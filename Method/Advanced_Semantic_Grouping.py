# -*- coding: utf-8 -*-
"""
Advanced Semantic Grouping với AI-driven optimizations
======================================================
- Multi-objective optimization (coherence + size + coverage)
- Dynamic similarity computation với context awareness
- Advanced group expansion strategies
- Quality-driven post-processing
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Set
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import minimize
import networkx as nx
from collections import defaultdict
import gc

class AdvancedSemanticGrouping:
    """Advanced semantic grouping với AI-driven optimizations"""
    
    def __init__(self, 
                 embedding_model: str = "thenlper/gte-base",
                 device: str = "auto"):
        self.embedding_model = embedding_model
        self.device = device
        self.grouping_history = []
        
    def multi_objective_grouping(self,
                               sentences: List[str],
                               embeddings: np.ndarray,
                               target_coherence: float = 0.7,
                               target_size_range: Tuple[int, int] = (3, 8),
                               weight_coherence: float = 0.4,
                               weight_size: float = 0.3,
                               weight_coverage: float = 0.3) -> List[List[int]]:
        """
        Multi-objective optimization cho semantic grouping:
        - Maximize intra-group coherence
        - Optimize group size distribution
        - Ensure complete coverage
        - Minimize inter-group similarity
        """
        
        if len(sentences) <= 3:
            return [list(range(len(sentences)))]
        
        # Build similarity graph
        sim_matrix = cosine_similarity(embeddings)
        similarity_graph = self._build_similarity_graph(sim_matrix, threshold=0.3)
        
        # Find optimal grouping using multi-objective optimization
        best_grouping = self._optimize_grouping(
            similarity_graph, sim_matrix, len(sentences),
            target_coherence, target_size_range,
            weight_coherence, weight_size, weight_coverage
        )
        
        return best_grouping
    
    def _build_similarity_graph(self, 
                              sim_matrix: np.ndarray, 
                              threshold: float) -> nx.Graph:
        """Build weighted graph from similarity matrix"""
        G = nx.Graph()
        n = sim_matrix.shape[0]
        
        # Add nodes
        for i in range(n):
            G.add_node(i)
        
        # Add edges for similarities above threshold
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=sim_matrix[i, j])
        
        return G
    
    def _optimize_grouping(self,
                         graph: nx.Graph,
                         sim_matrix: np.ndarray,
                         n_sentences: int,
                         target_coherence: float,
                         target_size_range: Tuple[int, int],
                         w_coherence: float,
                         w_size: float,
                         w_coverage: float) -> List[List[int]]:
        """
        Optimize grouping using graph algorithms + heuristics
        """
        
        # Start with community detection as initial solution
        try:
            import networkx.algorithms.community as nx_comm
            communities = list(nx_comm.greedy_modularity_communities(graph))
            initial_grouping = [list(community) for community in communities]
        except:
            # Fallback: simple connected components
            initial_grouping = [list(component) for component in nx.connected_components(graph)]
        
        if not initial_grouping:
            # Fallback: sequential grouping
            return self._sequential_grouping(sim_matrix, target_size_range)
        
        # Refine grouping using local optimization
        refined_grouping = self._refine_grouping(
            initial_grouping, sim_matrix, target_coherence, target_size_range
        )
        
        return refined_grouping
    
    def _sequential_grouping(self, 
                           sim_matrix: np.ndarray,
                           target_size_range: Tuple[int, int]) -> List[List[int]]:
        """Fallback sequential grouping algorithm"""
        n = sim_matrix.shape[0]
        ungrouped = set(range(n))
        groups = []
        min_size, max_size = target_size_range
        
        while ungrouped:
            # Start new group with highest remaining similarity pair
            if len(ungrouped) == 1:
                groups.append(list(ungrouped))
                break
            
            best_pair = None
            best_sim = -1
            
            for i in ungrouped:
                for j in ungrouped:
                    if i < j and sim_matrix[i, j] > best_sim:
                        best_sim = sim_matrix[i, j]
                        best_pair = (i, j)
            
            if best_pair is None:
                # No good pairs, group remaining individually
                groups.extend([[idx] for idx in ungrouped])
                break
            
            # Start group with best pair
            current_group = [best_pair[0], best_pair[1]]
            ungrouped.remove(best_pair[0])
            ungrouped.remove(best_pair[1])
            
            # Expand group greedily
            while len(current_group) < max_size and ungrouped:
                best_candidate = None
                best_avg_sim = -1
                
                for candidate in ungrouped:
                    # Calculate average similarity to current group
                    avg_sim = np.mean([sim_matrix[candidate, member] for member in current_group])
                    if avg_sim > best_avg_sim:
                        best_avg_sim = avg_sim
                        best_candidate = candidate
                
                # Add if similarity is good enough
                if best_candidate is not None and best_avg_sim > 0.3:
                    current_group.append(best_candidate)
                    ungrouped.remove(best_candidate)
                else:
                    break
            
            groups.append(current_group)
        
        return groups
    
    def _refine_grouping(self,
                       initial_grouping: List[List[int]],
                       sim_matrix: np.ndarray,
                       target_coherence: float,
                       target_size_range: Tuple[int, int]) -> List[List[int]]:
        """Refine grouping using local search and optimization"""
        
        grouping = [group.copy() for group in initial_grouping]
        min_size, max_size = target_size_range
        improved = True
        iterations = 0
        max_iterations = 5
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Merge small groups
            grouping = self._merge_small_groups(grouping, sim_matrix, min_size)
            
            # Split large groups
            grouping = self._split_large_groups(grouping, sim_matrix, max_size)
            
            # Reassign poorly fitting sentences
            new_grouping = self._reassign_outliers(grouping, sim_matrix, target_coherence)
            
            if new_grouping != grouping:
                improved = True
                grouping = new_grouping
        
        return grouping
    
    def _merge_small_groups(self,
                          grouping: List[List[int]],
                          sim_matrix: np.ndarray,
                          min_size: int) -> List[List[int]]:
        """Merge groups that are too small"""
        small_groups = [i for i, group in enumerate(grouping) if len(group) < min_size]
        
        if not small_groups:
            return grouping
        
        new_grouping = []
        merged_indices = set()
        
        for i, group in enumerate(grouping):
            if i in merged_indices:
                continue
                
            if i in small_groups:
                # Find best group to merge with
                best_merge_idx = None
                best_merge_score = -1
                
                for j, other_group in enumerate(grouping):
                    if j != i and j not in merged_indices and j not in small_groups:
                        # Calculate merge compatibility
                        merge_score = self._calculate_merge_score(group, other_group, sim_matrix)
                        if merge_score > best_merge_score:
                            best_merge_score = merge_score
                            best_merge_idx = j
                
                if best_merge_idx is not None:
                    # Merge groups
                    merged_group = group + grouping[best_merge_idx]
                    new_grouping.append(merged_group)
                    merged_indices.add(i)
                    merged_indices.add(best_merge_idx)
                else:
                    # Keep as is if no good merge found
                    new_grouping.append(group)
            else:
                if i not in merged_indices:
                    new_grouping.append(group)
        
        return new_grouping
    
    def _split_large_groups(self,
                          grouping: List[List[int]],
                          sim_matrix: np.ndarray,
                          max_size: int) -> List[List[int]]:
        """Split groups that are too large"""
        new_grouping = []
        
        for group in grouping:
            if len(group) <= max_size:
                new_grouping.append(group)
            else:
                # Split using spectral clustering or k-means
                sub_groups = self._split_group_optimally(group, sim_matrix, max_size)
                new_grouping.extend(sub_groups)
        
        return new_grouping
    
    def _split_group_optimally(self,
                             group: List[int],
                             sim_matrix: np.ndarray,
                             max_size: int) -> List[List[int]]:
        """Split a large group into optimal sub-groups"""
        if len(group) <= max_size:
            return [group]
        
        # Extract sub-matrix for this group
        group_sim = sim_matrix[np.ix_(group, group)]
        
        # Find optimal split point using spectral analysis
        try:
            from sklearn.cluster import SpectralClustering
            
            n_clusters = (len(group) + max_size - 1) // max_size  # Ceiling division
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            
            cluster_labels = clustering.fit_predict(group_sim)
            
            # Group by cluster labels
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(group[i])
            
            return list(clusters.values())
            
        except:
            # Fallback: simple binary split
            mid = len(group) // 2
            return [group[:mid], group[mid:]]
    
    def _reassign_outliers(self,
                         grouping: List[List[int]],
                         sim_matrix: np.ndarray,
                         target_coherence: float) -> List[List[int]]:
        """Reassign sentences that don't fit well in their current group"""
        new_grouping = [group.copy() for group in grouping]
        
        for group_idx, group in enumerate(grouping):
            if len(group) <= 1:
                continue
            
            outliers = []
            for sentence_idx in group:
                # Calculate coherence with rest of group
                other_sentences = [s for s in group if s != sentence_idx]
                if not other_sentences:
                    continue
                
                similarities = [sim_matrix[sentence_idx, other] for other in other_sentences]
                avg_coherence = np.mean(similarities)
                
                if avg_coherence < target_coherence:
                    outliers.append(sentence_idx)
            
            # Reassign outliers to better groups
            for outlier in outliers:
                best_group_idx = None
                best_coherence = -1
                
                for other_group_idx, other_group in enumerate(new_grouping):
                    if other_group_idx == group_idx:
                        continue
                    
                    if not other_group:
                        continue
                    
                    # Calculate potential coherence with other group
                    similarities = [sim_matrix[outlier, member] for member in other_group]
                    avg_coherence = np.mean(similarities)
                    
                    if avg_coherence > best_coherence and avg_coherence > target_coherence:
                        best_coherence = avg_coherence
                        best_group_idx = other_group_idx
                
                # Reassign if better group found
                if best_group_idx is not None:
                    new_grouping[group_idx].remove(outlier)
                    new_grouping[best_group_idx].append(outlier)
        
        # Remove empty groups
        new_grouping = [group for group in new_grouping if group]
        
        return new_grouping
    
    def _calculate_merge_score(self,
                             group1: List[int],
                             group2: List[int],
                             sim_matrix: np.ndarray) -> float:
        """Calculate compatibility score for merging two groups"""
        if not group1 or not group2:
            return -1
        
        # Calculate inter-group similarity
        similarities = []
        for s1 in group1:
            for s2 in group2:
                similarities.append(sim_matrix[s1, s2])
        
        return np.mean(similarities) if similarities else -1
    
    def adaptive_threshold_evolution(self,
                                   sentences: List[str],
                                   embeddings: np.ndarray,
                                   initial_threshold: float = 0.5,
                                   max_iterations: int = 10) -> Tuple[List[List[int]], float]:
        """
        Evolve threshold adaptively to achieve optimal grouping quality
        """
        best_grouping = None
        best_quality = -1
        best_threshold = initial_threshold
        
        # Test different thresholds
        threshold_candidates = np.linspace(0.2, 0.8, 15)
        
        for threshold in threshold_candidates:
            try:
                sim_matrix = cosine_similarity(embeddings)
                grouping = self.multi_objective_grouping(
                    sentences, embeddings,
                    target_coherence=threshold
                )
                
                # Evaluate quality
                quality = self._evaluate_grouping_quality(grouping, sim_matrix)
                
                if quality > best_quality:
                    best_quality = quality
                    best_grouping = grouping
                    best_threshold = threshold
                    
            except Exception as e:
                continue
        
        return best_grouping or [[]], best_threshold
    
    def _evaluate_grouping_quality(self,
                                 grouping: List[List[int]],
                                 sim_matrix: np.ndarray) -> float:
        """Comprehensive quality evaluation"""
        if not grouping:
            return 0
        
        coherence_scores = []
        size_scores = []
        
        for group in grouping:
            if len(group) < 2:
                coherence_scores.append(1.0)  # Single sentence = perfect coherence
                size_scores.append(0.5)  # Penalize very small groups
                continue
            
            # Intra-group coherence
            group_sims = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    group_sims.append(sim_matrix[group[i], group[j]])
            
            coherence = np.mean(group_sims) if group_sims else 0
            coherence_scores.append(coherence)
            
            # Size score (prefer groups of size 3-7)
            ideal_size = 5
            size_penalty = abs(len(group) - ideal_size) / ideal_size
            size_score = max(0, 1 - size_penalty)
            size_scores.append(size_score)
        
        # Overall quality
        avg_coherence = np.mean(coherence_scores)
        avg_size_score = np.mean(size_scores)
        coverage = sum(len(group) for group in grouping) / sim_matrix.shape[0]
        
        return (avg_coherence * 0.5 + avg_size_score * 0.3 + coverage * 0.2)
