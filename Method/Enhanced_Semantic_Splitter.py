# -*- coding: utf-8 -*-
"""
Enhanced Semantic Splitter với các cải tiến mới
===============================================
- Adaptive semantic threshold dựa trên content complexity
- Multi-level chunking hierarchy (sentence -> paragraph -> document)
- Dynamic token balancing với quality preservation
- Advanced overlap strategies
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import gc

class EnhancedSemanticSplitter:
    """Enhanced semantic splitter với advanced features"""
    
    def __init__(self, 
                 embedding_model: str = "thenlper/gte-base",
                 device: str = "auto"):
        self.embedding_model = embedding_model
        self.device = device
        self.quality_metrics = {}
    
    def adaptive_threshold_selection(self, 
                                   embeddings: np.ndarray,
                                   target_chunks: int = None) -> float:
        """
        Tự động chọn semantic threshold dựa trên:
        - Content diversity (standard deviation of similarities)
        - Target number of chunks
        - Embedding quality score
        """
        if len(embeddings) < 2:
            return 0.3
            
        # Tính similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        
        # Lấy upper triangle (không bao gồm diagonal)
        upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        
        # Content diversity metrics
        mean_sim = np.mean(upper_tri)
        std_sim = np.std(upper_tri)
        
        # Adaptive threshold
        if std_sim < 0.1:  # Highly coherent content
            base_threshold = mean_sim - 0.1
        elif std_sim > 0.3:  # Highly diverse content  
            base_threshold = mean_sim + 0.1
        else:  # Moderately diverse
            base_threshold = mean_sim
            
        # Adjust based on target chunks
        if target_chunks:
            estimated_cuts = len(embeddings) / target_chunks
            if estimated_cuts > len(embeddings) * 0.8:  # Too many cuts
                base_threshold += 0.1
            elif estimated_cuts < len(embeddings) * 0.2:  # Too few cuts
                base_threshold -= 0.1
        
        return max(0.1, min(0.9, base_threshold))
    
    def hierarchical_chunking(self,
                            sentences: List[str],
                            embeddings: np.ndarray,
                            min_chunk_size: int = 50,
                            max_chunk_size: int = 500,
                            quality_threshold: float = 0.7) -> List[List[int]]:
        """
        Multi-level hierarchical chunking:
        1. Sentence-level semantic grouping
        2. Paragraph-level consolidation  
        3. Document-level optimization
        """
        if len(sentences) <= 3:
            return [list(range(len(sentences)))]
        
        # Level 1: Sentence clustering
        n_clusters = max(2, min(len(sentences) // 3, 10))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Level 2: Group by clusters and refine
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        # Level 3: Size and quality optimization
        final_chunks = []
        for cluster_indices in clusters.values():
            if not cluster_indices:
                continue
                
            chunk_embeddings = embeddings[cluster_indices]
            chunk_sentences = [sentences[i] for i in cluster_indices]
            
            # Check chunk quality
            if len(chunk_embeddings) > 1:
                cluster_sim = cosine_similarity(chunk_embeddings)
                avg_similarity = np.mean(cluster_sim[np.triu_indices_from(cluster_sim, k=1)])
                
                if avg_similarity < quality_threshold:
                    # Split low-quality cluster
                    sub_chunks = self._split_low_quality_cluster(
                        cluster_indices, embeddings, sentences, min_chunk_size
                    )
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(cluster_indices)
            else:
                final_chunks.append(cluster_indices)
        
        return final_chunks
    
    def _split_low_quality_cluster(self, 
                                 indices: List[int],
                                 embeddings: np.ndarray,
                                 sentences: List[str],
                                 min_size: int) -> List[List[int]]:
        """Split cluster với quality thấp thành sub-chunks"""
        if len(indices) <= min_size:
            return [indices]
        
        # Binary split based on maximum dissimilarity
        cluster_emb = embeddings[indices]
        sim_matrix = cosine_similarity(cluster_emb)
        
        # Find most dissimilar pair
        min_sim_idx = np.unravel_index(np.argmin(sim_matrix + np.eye(len(indices))), 
                                     sim_matrix.shape)
        
        # Split into two groups around these centroids
        centroid1 = cluster_emb[min_sim_idx[0]]
        centroid2 = cluster_emb[min_sim_idx[1]]
        
        group1, group2 = [], []
        for i, emb in enumerate(cluster_emb):
            sim1 = cosine_similarity([emb], [centroid1])[0, 0]
            sim2 = cosine_similarity([emb], [centroid2])[0, 0]
            
            if sim1 >= sim2:
                group1.append(indices[i])
            else:
                group2.append(indices[i])
        
        return [group1, group2] if len(group1) > 0 and len(group2) > 0 else [indices]
    
    def intelligent_overlap(self,
                          chunks: List[List[int]],
                          sentences: List[str],
                          embeddings: np.ndarray,
                          overlap_strategy: str = "semantic") -> List[List[int]]:
        """
        Intelligent overlap based on:
        - Semantic continuity
        - Topic transition detection
        - Information preservation
        """
        if overlap_strategy == "none" or len(chunks) <= 1:
            return chunks
        
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                enhanced_chunks.append(chunk)
                continue
            
            prev_chunk = chunks[i-1]
            
            if overlap_strategy == "semantic":
                # Find best overlap sentences based on semantic bridge
                overlap_size = self._calculate_semantic_overlap(
                    prev_chunk, chunk, embeddings
                )
                
                if overlap_size > 0:
                    overlap_indices = prev_chunk[-overlap_size:]
                    enhanced_chunk = overlap_indices + chunk
                else:
                    enhanced_chunk = chunk
            else:
                # Fixed overlap
                overlap_size = min(2, len(prev_chunk))
                overlap_indices = prev_chunk[-overlap_size:]
                enhanced_chunk = overlap_indices + chunk
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _calculate_semantic_overlap(self,
                                  prev_chunk: List[int],
                                  curr_chunk: List[int],
                                  embeddings: np.ndarray) -> int:
        """Calculate optimal overlap size based on semantic continuity"""
        if len(prev_chunk) < 2 or len(curr_chunk) < 2:
            return 1
        
        # Calculate transition similarity between chunks
        prev_end = embeddings[prev_chunk[-2:]]  # Last 2 sentences of previous
        curr_start = embeddings[curr_chunk[:2]]  # First 2 sentences of current
        
        transition_sim = cosine_similarity(prev_end, curr_start)
        max_sim = np.max(transition_sim)
        
        # Determine overlap based on transition strength
        if max_sim > 0.8:  # Strong semantic continuity
            return min(3, len(prev_chunk))
        elif max_sim > 0.6:  # Moderate continuity
            return min(2, len(prev_chunk))
        elif max_sim > 0.4:  # Weak continuity
            return 1
        else:  # No continuity
            return 0
    
    def quality_assessment(self,
                         chunks: List[List[int]],
                         sentences: List[str],
                         embeddings: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive quality assessment:
        - Intra-chunk coherence
        - Inter-chunk separation
        - Size distribution balance
        - Information coverage
        """
        if not chunks:
            return {"error": "No chunks to assess"}
        
        # Intra-chunk coherence
        coherence_scores = []
        for chunk in chunks:
            if len(chunk) > 1:
                chunk_emb = embeddings[chunk]
                sim_matrix = cosine_similarity(chunk_emb)
                avg_coherence = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
                coherence_scores.append(avg_coherence)
        
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0
        
        # Inter-chunk separation
        separation_scores = []
        for i in range(len(chunks) - 1):
            chunk1_emb = embeddings[chunks[i]]
            chunk2_emb = embeddings[chunks[i + 1]]
            
            # Average similarity between chunks
            inter_sim = cosine_similarity(chunk1_emb, chunk2_emb)
            avg_separation = 1 - np.mean(inter_sim)  # Higher = better separation
            separation_scores.append(avg_separation)
        
        avg_separation = np.mean(separation_scores) if separation_scores else 0
        
        # Size distribution
        chunk_sizes = [len(chunk) for chunk in chunks]
        size_std = np.std(chunk_sizes)
        size_balance = 1 / (1 + size_std)  # Lower std = better balance
        
        # Information coverage (no sentence left behind)
        total_sentences = sum(len(chunk) for chunk in chunks)
        coverage = total_sentences / len(sentences)
        
        return {
            "coherence": avg_coherence,
            "separation": avg_separation,
            "size_balance": size_balance,
            "coverage": coverage,
            "overall_quality": (avg_coherence + avg_separation + size_balance + coverage) / 4
        }
