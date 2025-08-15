# -*- coding: utf-8 -*-
"""
Unified Chunking Framework với A/B Testing và Performance Monitoring
====================================================================
- Real-time performance comparison
- Automated parameter optimization
- Quality-driven model selection
- Comprehensive evaluation metrics
"""

import time
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from Enhanced_Semantic_Splitter import EnhancedSemanticSplitter
from Advanced_Semantic_Grouping import AdvancedSemanticGrouping

@dataclass
class ChunkingResult:
    """Results from a chunking method"""
    method_name: str
    chunks: List[Tuple[str, str, Optional[str]]]  # (id, text, oie)
    processing_time: float
    quality_metrics: Dict[str, float]
    parameter_config: Dict[str, Any]
    memory_usage: float

@dataclass
class ComparisonReport:
    """Comprehensive comparison report"""
    timestamp: str
    input_stats: Dict[str, Any]
    method_results: List[ChunkingResult]
    winner: str
    performance_summary: Dict[str, float]
    recommendations: List[str]

class UnifiedChunkingFramework:
    """Unified framework for chunking comparison and optimization"""
    
    def __init__(self, output_dir: str = "./chunking_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize chunking methods
        self.enhanced_splitter = EnhancedSemanticSplitter()
        self.advanced_grouping = AdvancedSemanticGrouping()
        
        # Performance tracking
        self.performance_history = []
        self.optimization_cache = {}
        
    def comprehensive_evaluation(self,
                                doc_id: str,
                                passage_text: str,
                                target_metrics: Dict[str, float] = None,
                                include_oie: bool = False) -> ComparisonReport:
        """
        Comprehensive evaluation of all chunking methods
        
        Args:
            doc_id: Document identifier
            passage_text: Text to chunk
            target_metrics: Target quality thresholds
            include_oie: Whether to include OIE extraction
            
        Returns:
            ComparisonReport with detailed comparison
        """
        if target_metrics is None:
            target_metrics = {
                "coherence": 0.7,
                "coverage": 0.95,
                "size_balance": 0.8,
                "processing_speed": 2.0  # chunks per second
            }
        
        print(f"🔍 Starting comprehensive evaluation for {doc_id}")
        print(f"📄 Text length: {len(passage_text)} characters")
        
        # Extract sentences and embeddings (shared across methods)
        sentences = self._extract_sentences(passage_text)
        embeddings = self._get_embeddings(sentences)
        
        input_stats = {
            "doc_id": doc_id,
            "text_length": len(passage_text),
            "sentence_count": len(sentences),
            "avg_sentence_length": np.mean([len(s) for s in sentences]) if sentences else 0
        }
        
        # Test configurations for each method
        test_configs = self._generate_test_configurations()
        
        # Run all methods in parallel
        method_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            # Enhanced Splitter configurations
            for config in test_configs["enhanced_splitter"]:
                future = executor.submit(
                    self._test_enhanced_splitter,
                    doc_id, sentences, embeddings, config, include_oie
                )
                futures[future] = f"enhanced_splitter_{config['name']}"
            
            # Advanced Grouping configurations
            for config in test_configs["advanced_grouping"]:
                future = executor.submit(
                    self._test_advanced_grouping,
                    doc_id, sentences, embeddings, config, include_oie
                )
                futures[future] = f"advanced_grouping_{config['name']}"
            
            # Legacy methods for comparison
            for config in test_configs["legacy"]:
                future = executor.submit(
                    self._test_legacy_method,
                    doc_id, passage_text, config, include_oie
                )
                futures[future] = f"legacy_{config['name']}"
            
            # Collect results
            for future in as_completed(futures):
                method_name = futures[future]
                try:
                    result = future.result()
                    if result:
                        method_results.append(result)
                        print(f"Completed {method_name}: {len(result.chunks)} chunks")
                except Exception as e:
                    print(f"Failed {method_name}: {e}")
        
        # Analyze and rank results
        performance_summary = self._analyze_results(method_results, target_metrics)
        winner = self._select_winner(method_results, target_metrics)
        recommendations = self._generate_recommendations(method_results, target_metrics)
        
        # Create comprehensive report
        report = ComparisonReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            input_stats=input_stats,
            method_results=method_results,
            winner=winner,
            performance_summary=performance_summary,
            recommendations=recommendations
        )
        
        # Save report
        self._save_report(report, doc_id)
        
        return report
    
    def _generate_test_configurations(self) -> Dict[str, List[Dict]]:
        """Generate test configurations for all methods"""
        return {
            "enhanced_splitter": [
                {
                    "name": "adaptive_high_quality",
                    "params": {
                        "target_chunks": 8,
                        "quality_threshold": 0.8,
                        "overlap_strategy": "semantic"
                    }
                },
                {
                    "name": "balanced_performance",
                    "params": {
                        "target_chunks": 6,
                        "quality_threshold": 0.7,
                        "overlap_strategy": "semantic"
                    }
                },
                {
                    "name": "fast_processing",
                    "params": {
                        "target_chunks": 5,
                        "quality_threshold": 0.6,
                        "overlap_strategy": "fixed"
                    }
                }
            ],
            "advanced_grouping": [
                {
                    "name": "multi_objective_optimal",
                    "params": {
                        "target_coherence": 0.75,
                        "target_size_range": (3, 7),
                        "weight_coherence": 0.5,
                        "weight_size": 0.3,
                        "weight_coverage": 0.2
                    }
                },
                {
                    "name": "coherence_focused",
                    "params": {
                        "target_coherence": 0.8,
                        "target_size_range": (2, 6),
                        "weight_coherence": 0.7,
                        "weight_size": 0.2,
                        "weight_coverage": 0.1
                    }
                },
                {
                    "name": "size_balanced",
                    "params": {
                        "target_coherence": 0.65,
                        "target_size_range": (4, 8),
                        "weight_coherence": 0.3,
                        "weight_size": 0.5,
                        "weight_coverage": 0.2
                    }
                }
            ],
            "legacy": [
                {
                    "name": "semantic_splitter_baseline",
                    "method": "semantic_splitter",
                    "params": {
                        "semantic_threshold": 0.3,
                        "chunk_overlap": 0
                    }
                },
                {
                    "name": "semantic_grouping_baseline",
                    "method": "semantic_grouping",
                    "params": {
                        "initial_threshold": "auto",
                        "decay_factor": 0.85,
                        "min_threshold": "auto"
                    }
                }
            ]
        }
    
    def _test_enhanced_splitter(self,
                              doc_id: str,
                              sentences: List[str],
                              embeddings: np.ndarray,
                              config: Dict,
                              include_oie: bool) -> ChunkingResult:
        """Test Enhanced Splitter with specific configuration"""
        import psutil
        import os
        
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Adaptive threshold selection
            threshold = self.enhanced_splitter.adaptive_threshold_selection(
                embeddings, config["params"].get("target_chunks")
            )
            
            # Hierarchical chunking
            chunk_indices = self.enhanced_splitter.hierarchical_chunking(
                sentences, embeddings,
                quality_threshold=config["params"]["quality_threshold"]
            )
            
            # Intelligent overlap
            chunk_indices = self.enhanced_splitter.intelligent_overlap(
                chunk_indices, sentences, embeddings,
                config["params"]["overlap_strategy"]
            )
            
            # Quality assessment
            quality_metrics = self.enhanced_splitter.quality_assessment(
                chunk_indices, sentences, embeddings
            )
            
            # Create final chunks
            chunks = []
            for i, indices in enumerate(chunk_indices):
                chunk_text = " ".join([sentences[idx] for idx in indices])
                chunk_id = f"{doc_id}_enhanced_split_{i}"
                
                oie_info = None
                if include_oie:
                    oie_info = self._extract_oie(chunk_text)
                
                chunks.append((chunk_id, chunk_text, oie_info))
            
        except Exception as e:
            print(f"Error in enhanced splitter: {e}")
            return None
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        return ChunkingResult(
            method_name=f"enhanced_splitter_{config['name']}",
            chunks=chunks,
            processing_time=end_time - start_time,
            quality_metrics=quality_metrics,
            parameter_config=config["params"],
            memory_usage=end_memory - start_memory
        )
    
    def _test_advanced_grouping(self,
                              doc_id: str,
                              sentences: List[str],
                              embeddings: np.ndarray,
                              config: Dict,
                              include_oie: bool) -> ChunkingResult:
        """Test Advanced Grouping with specific configuration"""
        import psutil
        import os
        
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Multi-objective grouping
            chunk_indices = self.advanced_grouping.multi_objective_grouping(
                sentences, embeddings, **config["params"]
            )
            
            # Quality evaluation
            sim_matrix = cosine_similarity(embeddings)
            quality_score = self.advanced_grouping._evaluate_grouping_quality(
                chunk_indices, sim_matrix
            )
            
            quality_metrics = {
                "overall_quality": quality_score,
                "group_count": len(chunk_indices),
                "avg_group_size": np.mean([len(group) for group in chunk_indices]),
                "size_std": np.std([len(group) for group in chunk_indices])
            }
            
            # Create final chunks
            chunks = []
            for i, indices in enumerate(chunk_indices):
                chunk_text = " ".join([sentences[idx] for idx in indices])
                chunk_id = f"{doc_id}_advanced_group_{i}"
                
                oie_info = None
                if include_oie:
                    oie_info = self._extract_oie(chunk_text)
                
                chunks.append((chunk_id, chunk_text, oie_info))
        
        except Exception as e:
            print(f"Error in advanced grouping: {e}")
            return None
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        return ChunkingResult(
            method_name=f"advanced_grouping_{config['name']}",
            chunks=chunks,
            processing_time=end_time - start_time,
            quality_metrics=quality_metrics,
            parameter_config=config["params"],
            memory_usage=end_memory - start_memory
        )
    
    def _test_legacy_method(self,
                          doc_id: str,
                          passage_text: str,
                          config: Dict,
                          include_oie: bool) -> ChunkingResult:
        """Test legacy methods for comparison"""
        # Implementation would call existing methods
        # This is a placeholder for integration with existing codebase
        pass
    
    def _analyze_results(self,
                       results: List[ChunkingResult],
                       target_metrics: Dict[str, float]) -> Dict[str, float]:
        """Analyze and compare all results"""
        analysis = {
            "total_methods_tested": len(results),
            "avg_processing_time": np.mean([r.processing_time for r in results]),
            "avg_memory_usage": np.mean([r.memory_usage for r in results]),
            "quality_leader": "",
            "speed_leader": "",
            "memory_efficient": ""
        }
        
        if results:
            # Find leaders in different categories
            quality_scores = []
            for result in results:
                if "overall_quality" in result.quality_metrics:
                    quality_scores.append((result.method_name, result.quality_metrics["overall_quality"]))
            
            if quality_scores:
                analysis["quality_leader"] = max(quality_scores, key=lambda x: x[1])[0]
            
            analysis["speed_leader"] = min(results, key=lambda x: x.processing_time).method_name
            analysis["memory_efficient"] = min(results, key=lambda x: x.memory_usage).method_name
        
        return analysis
    
    def _select_winner(self,
                     results: List[ChunkingResult],
                     target_metrics: Dict[str, float]) -> str:
        """Select overall winner based on weighted criteria"""
        if not results:
            return "No valid results"
        
        # Weighted scoring system
        weights = {
            "quality": 0.4,
            "speed": 0.3,
            "memory": 0.2,
            "coverage": 0.1
        }
        
        scores = {}
        for result in results:
            score = 0
            
            # Quality score
            quality = result.quality_metrics.get("overall_quality", 0)
            score += weights["quality"] * quality
            
            # Speed score (inverse of processing time)
            max_time = max([r.processing_time for r in results])
            speed_score = 1 - (result.processing_time / max_time) if max_time > 0 else 1
            score += weights["speed"] * speed_score
            
            # Memory efficiency (inverse of memory usage)
            max_memory = max([r.memory_usage for r in results])
            memory_score = 1 - (result.memory_usage / max_memory) if max_memory > 0 else 1
            score += weights["memory"] * memory_score
            
            # Coverage score
            coverage = result.quality_metrics.get("coverage", 1.0)
            score += weights["coverage"] * coverage
            
            scores[result.method_name] = score
        
        return max(scores, key=scores.get) if scores else "No winner determined"
    
    def _generate_recommendations(self,
                                results: List[ChunkingResult],
                                target_metrics: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if not results:
            return ["No results to analyze. Check input data and method implementations."]
        
        # Analyze performance patterns
        quality_scores = [r.quality_metrics.get("overall_quality", 0) for r in results]
        processing_times = [r.processing_time for r in results]
        
        avg_quality = np.mean(quality_scores)
        avg_time = np.mean(processing_times)
        
        # Quality recommendations
        if avg_quality < target_metrics.get("coherence", 0.7):
            recommendations.append(
                "⚠️  Overall quality below target. Consider using stricter similarity thresholds "
                "or increasing minimum sentences per chunk."
            )
        
        # Performance recommendations
        if avg_time > target_metrics.get("processing_speed", 2.0):
            recommendations.append(
                "🚀 Processing speed can be improved. Consider parallel processing or "
                "reducing embedding dimensions."
            )
        
        # Method-specific recommendations
        advanced_methods = [r for r in results if "advanced" in r.method_name.lower()]
        enhanced_methods = [r for r in results if "enhanced" in r.method_name.lower()]
        
        if advanced_methods and enhanced_methods:
            best_advanced = max(advanced_methods, key=lambda x: x.quality_metrics.get("overall_quality", 0))
            best_enhanced = max(enhanced_methods, key=lambda x: x.quality_metrics.get("overall_quality", 0))
            
            if best_advanced.quality_metrics.get("overall_quality", 0) > best_enhanced.quality_metrics.get("overall_quality", 0):
                recommendations.append(
                    "🎯 Advanced Grouping shows superior quality. Recommended for high-quality chunking needs."
                )
            else:
                recommendations.append(
                    "⚡ Enhanced Splitter provides better balance of quality and performance."
                )
        
        # Configuration recommendations
        recommendations.append(
            "🔧 For production use, consider A/B testing with top 2-3 configurations "
            "on your specific dataset."
        )
        
        return recommendations
    
    def _save_report(self, report: ComparisonReport, doc_id: str):
        """Save comprehensive report"""
        timestamp = report.timestamp.replace(":", "-").replace(" ", "_")
        report_file = self.output_dir / f"chunking_report_{doc_id}_{timestamp}.json"
        
        # Convert to serializable format
        report_dict = asdict(report)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 Comprehensive report saved: {report_file}")
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences using spacy"""
        # Implementation would use existing sentence extraction
        # Placeholder for integration
        return text.split('. ')
    
    def _get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get embeddings for sentences"""
        # Implementation would use existing embedding generation
        # Placeholder for integration
        return np.random.random((len(sentences), 768))  # Dummy embeddings
    
    def _extract_oie(self, text: str) -> Optional[str]:
        """Extract Open Information Extraction"""
        # Implementation would use existing OIE extraction
        # Placeholder for integration
        return None
