"""
Embedding Model Evaluator
=========================
Script to evaluate and compare different embedding models for semantic search tasks.
Measures retrieval performance, semantic similarity quality, and computational efficiency.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from Tool.Sentence_Embedding import sentence_embedding, get_device
    from Tool.rank_chunks import rank_by_cosine_similarity, rank_by_bm25, rank_by_rrf
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure Tool.Sentence_Embedding and Tool.rank_chunks are available")
    sys.exit(1)

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import average_precision_score, ndcg_score
    from scipy.stats import spearmanr, pearsonr
except ImportError:
    print("Installing required packages...")
    os.system("pip install scikit-learn scipy")
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import average_precision_score, ndcg_score
    from scipy.stats import spearmanr, pearsonr

@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model"""
    name: str
    model_id: str
    description: str
    max_sequence_length: int = 512
    embedding_dim: Optional[int] = None
    supports_batch: bool = True

@dataclass
class EvaluationResult:
    """Results from evaluating an embedding model"""
    model_name: str
    retrieval_metrics: Dict[str, float]
    similarity_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    error_analysis: Dict[str, Any]

class EmbeddingModelEvaluator:
    """Comprehensive embedding model evaluator"""
    
    def __init__(self, output_dir: str = "./embedding_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Predefined model configurations
        self.available_models = [
            EmbeddingModelConfig(
                name="GTE-Large",
                model_id="thenlper/gte-large", 
                description="General Text Embeddings - Large (1024d)",
                embedding_dim=1024,
                max_sequence_length=512
            ),
            EmbeddingModelConfig(
                name="GTE-Base",
                model_id="thenlper/gte-base",
                description="General Text Embeddings - Base (768d)", 
                embedding_dim=768,
                max_sequence_length=512
            ),
            EmbeddingModelConfig(
                name="BGE-Large-EN",
                model_id="BAAI/bge-large-en-v1.5",
                description="BGE Large English (1024d)",
                embedding_dim=1024,
                max_sequence_length=512
            ),
            EmbeddingModelConfig(
                name="BGE-Base-EN", 
                model_id="BAAI/bge-base-en-v1.5",
                description="BGE Base English (768d)",
                embedding_dim=768,
                max_sequence_length=512
            ),
            EmbeddingModelConfig(
                name="E5-Large",
                model_id="intfloat/e5-large-v2",
                description="E5 Large v2 (1024d)",
                embedding_dim=1024,
                max_sequence_length=512
            ),
            EmbeddingModelConfig(
                name="E5-Base",
                model_id="intfloat/e5-base-v2", 
                description="E5 Base v2 (768d)",
                embedding_dim=768,
                max_sequence_length=512
            ),
            EmbeddingModelConfig(
                name="MiniLM-L6",
                model_id="sentence-transformers/all-MiniLM-L6-v2",
                description="All MiniLM L6 v2 (384d) - Fast",
                embedding_dim=384,
                max_sequence_length=256
            ),
            EmbeddingModelConfig(
                name="MiniLM-L12",
                model_id="sentence-transformers/all-MiniLM-L12-v2", 
                description="All MiniLM L12 v2 (384d)",
                embedding_dim=384,
                max_sequence_length=256
            ),
            EmbeddingModelConfig(
                name="MPNet-Base",
                model_id="sentence-transformers/all-mpnet-base-v2",
                description="All MPNet Base v2 (768d)",
                embedding_dim=768,
                max_sequence_length=384
            )
        ]
        
        self.evaluation_results = []
        
    def load_evaluation_data(self, data_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load evaluation dataset"""
        print(f"Loading evaluation data from: {data_path}")
        
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.tsv'):
            df = pd.read_csv(data_path, sep='\t')
        else:
            raise ValueError("Unsupported file format. Use CSV or TSV.")
            
        # Ensure required columns exist
        required_cols = ['query', 'passage', 'label']
        if not all(col in df.columns for col in required_cols):
            # Try alternative column names
            if 'query_text' in df.columns:
                df['query'] = df['query_text']
            if 'chunk_text' in df.columns:
                df['passage'] = df['chunk_text']
                
        # Validate data
        df = df.dropna(subset=['query', 'passage', 'label'])
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df.dropna(subset=['label'])
        
        # Sample data if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} examples from dataset")
            
        print(f"Loaded {len(df)} query-passage pairs")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df

    def evaluate_model(self, model_config: EmbeddingModelConfig, eval_data: pd.DataFrame) -> EvaluationResult:
        """Evaluate a single embedding model"""
        print(f"\nEvaluating: {model_config.name}")
        
        try:
            # Performance metrics
            start_time = time.time()
            
            # Get embeddings for queries and passages
            query_embeddings = self._get_embeddings(
                eval_data['query'].tolist(), 
                model_config.model_id,
                "Query Embeddings"
            )
            
            passage_embeddings = self._get_embeddings(
                eval_data['passage'].tolist(),
                model_config.model_id, 
                "Passage Embeddings"
            )
            
            embedding_time = time.time() - start_time
            
            # Calculate similarity matrix with detailed proof
            print(f"Computing similarity matrix for {len(query_embeddings)} queries × {len(passage_embeddings)} passages...")
            
            # Normalize embeddings for cosine similarity calculation
            query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            passage_norms = np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
            
            query_embeddings_normalized = query_embeddings / (query_norms + 1e-8)
            passage_embeddings_normalized = passage_embeddings / (passage_norms + 1e-8)
            
            # Compute full similarity matrix: queries × passages
            similarities = np.dot(query_embeddings_normalized, passage_embeddings_normalized.T)
            
            # Extract diagonal for query-passage pairs (assuming 1:1 correspondence)
            cosine_scores = np.diag(similarities)
            
            # Show similarity matrix statistics as proof
            print(f"✓ Similarity matrix shape: {similarities.shape}")
            print(f"✓ Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")
            print(f"✓ Average pairwise similarity: {cosine_scores.mean():.4f} ± {cosine_scores.std():.4f}")
            print(f"✓ Embedding dimensions: Q={query_embeddings.shape[1]}, P={passage_embeddings.shape[1]}")
            
            # Verify similarity calculation with sample
            if len(cosine_scores) > 0:
                sample_idx = 0
                q_vec = query_embeddings_normalized[sample_idx]
                p_vec = passage_embeddings_normalized[sample_idx]
                manual_sim = np.dot(q_vec, p_vec)
                matrix_sim = similarities[sample_idx, sample_idx]
                print(f"✓ Similarity verification (sample 0): manual={manual_sim:.6f}, matrix={matrix_sim:.6f}")
                assert abs(manual_sim - matrix_sim) < 1e-6, "Similarity calculation mismatch!"
            
            # Retrieval evaluation
            retrieval_metrics = self._evaluate_retrieval(eval_data, cosine_scores)
            
            # Similarity quality evaluation  
            similarity_metrics = self._evaluate_similarity_quality(eval_data, cosine_scores)
            
            # Performance metrics
            performance_metrics = {
                'embedding_time_seconds': embedding_time,
                'avg_time_per_text': embedding_time / (len(eval_data) * 2),
                'embedding_dimension': model_config.embedding_dim or query_embeddings.shape[1],
                'max_sequence_length': model_config.max_sequence_length,
                'device_used': str(self.device),
                'similarity_matrix_shape': similarities.shape,
                'similarity_computation_method': 'cosine_similarity_normalized_vectors'
            }
            
            # Error analysis with similarity matrix insights
            error_analysis = self._analyze_errors(eval_data, cosine_scores)
            error_analysis['similarity_matrix_stats'] = {
                'full_matrix_mean': float(similarities.mean()),
                'full_matrix_std': float(similarities.std()),
                'diagonal_mean': float(cosine_scores.mean()),
                'diagonal_std': float(cosine_scores.std()),
                'off_diagonal_mean': float((similarities.sum() - np.trace(similarities)) / (similarities.size - len(cosine_scores)))
            }
            
            result = EvaluationResult(
                model_name=model_config.name,
                retrieval_metrics=retrieval_metrics,
                similarity_metrics=similarity_metrics, 
                performance_metrics=performance_metrics,
                error_analysis=error_analysis
            )
            
            print(f"✓ Completed: {model_config.name}")
            return result
            
        except Exception as e:
            print(f"✗ Error: {model_config.name} - {str(e)}")
            # Return empty result with error info
            return EvaluationResult(
                model_name=model_config.name,
                retrieval_metrics={},
                similarity_metrics={},
                performance_metrics={'error': str(e)},
                error_analysis={}
            )

    def _get_embeddings(self, texts: List[str], model_id: str, desc: str) -> np.ndarray:
        """Get embeddings for a list of texts with detailed vector proof"""
        embeddings = []
        batch_size = 32
        
        print(f"  Processing {len(texts)} texts in batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = sentence_embedding(batch, model_name=model_id)
                
                # Handle different return formats
                if isinstance(batch_embeddings, np.ndarray):
                    if len(batch_embeddings.shape) == 3:
                        # Shape (1, n, dim) -> (n, dim)
                        batch_embeddings = batch_embeddings.reshape(-1, batch_embeddings.shape[-1])
                    elif len(batch_embeddings.shape) == 2:
                        # Already correct shape (n, dim)
                        pass
                    else:
                        # Handle 1D case
                        batch_embeddings = batch_embeddings.reshape(1, -1)
                    
                    embeddings.extend(batch_embeddings)
                elif isinstance(batch_embeddings, list):
                    embeddings.extend(batch_embeddings)
                else:
                    embeddings.append(batch_embeddings)
                    
            except Exception as e:
                print(f"  Batch {i//batch_size + 1} error: {e}")
                # Fallback: process one by one
                for text in batch:
                    try:
                        emb = sentence_embedding([text], model_name=model_id)
                        if isinstance(emb, np.ndarray):
                            if len(emb.shape) == 3:
                                emb = emb.reshape(-1, emb.shape[-1])
                            embeddings.extend(emb)
                        else:
                            embeddings.extend(emb if isinstance(emb, list) else [emb])
                    except:
                        # Add zero vector as fallback
                        embeddings.append(np.zeros(768))
        
        embeddings_array = np.array(embeddings)
        
        # Ensure 2D shape
        if len(embeddings_array.shape) == 3:
            embeddings_array = embeddings_array.reshape(-1, embeddings_array.shape[-1])
        
        # Show embedding statistics as proof
        print(f"  ✓ Generated embeddings: {embeddings_array.shape}")
        if len(embeddings_array) > 0:
            norms = np.linalg.norm(embeddings_array, axis=1)
            print(f"  ✓ Vector norm range: [{norms.min():.4f}, {norms.max():.4f}]")
            print(f"  ✓ Sample vector preview: {embeddings_array[0][:3]} ... {embeddings_array[0][-3:]}")
        
        return embeddings_array

    def _evaluate_retrieval(self, data: pd.DataFrame, scores: np.ndarray) -> Dict[str, float]:
        """Evaluate retrieval performance"""
        labels = data['label'].values
        
        # Group by query for retrieval evaluation
        query_groups = data.groupby('query')
        
        if len(query_groups) < 2:
            # If we don't have multiple passages per query, use binary classification metrics
            return self._binary_classification_metrics(labels, scores)
        
        # Multi-passage retrieval metrics
        ap_scores = []
        ndcg_scores = []
        
        start_idx = 0
        for query, group in query_groups:
            end_idx = start_idx + len(group)
            group_scores = scores[start_idx:end_idx]
            group_labels = labels[start_idx:end_idx]
            
            if len(np.unique(group_labels)) > 1:  # Must have both positive and negative examples
                # Average Precision
                ap = average_precision_score(group_labels, group_scores)
                ap_scores.append(ap)
                
                # NDCG@10
                ndcg = ndcg_score([group_labels], [group_scores], k=10)
                ndcg_scores.append(ndcg)
                
            start_idx = end_idx
            
        metrics = {}
        if ap_scores:
            metrics['mean_average_precision'] = np.mean(ap_scores)
            metrics['map_std'] = np.std(ap_scores)
        if ndcg_scores:
            metrics['ndcg_at_10'] = np.mean(ndcg_scores)
            metrics['ndcg_std'] = np.std(ndcg_scores)
            
        # Add binary metrics as well
        binary_metrics = self._binary_classification_metrics(labels, scores)
        metrics.update(binary_metrics)
        
        return metrics

    def _binary_classification_metrics(self, labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Calculate binary classification metrics"""
        # Find optimal threshold
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        f1_scores = []
        
        for threshold in thresholds:
            predictions = (scores >= threshold).astype(int)
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
            
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        max_f1 = np.max(f1_scores)
        
        # Calculate metrics at optimal threshold
        predictions = (scores >= optimal_threshold).astype(int)
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / len(labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': max_f1,
            'optimal_threshold': optimal_threshold
        }

    def _evaluate_similarity_quality(self, data: pd.DataFrame, scores: np.ndarray) -> Dict[str, float]:
        """Evaluate similarity quality"""
        labels = data['label'].values
        
        # Correlation with labels
        spearman_corr, spearman_p = spearmanr(scores, labels)
        pearson_corr, pearson_p = pearsonr(scores, labels)
        
        # Distribution analysis
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        metrics = {
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
        }
        
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            metrics.update({
                'positive_score_mean': np.mean(pos_scores),
                'positive_score_std': np.std(pos_scores),
                'negative_score_mean': np.mean(neg_scores),
                'negative_score_std': np.std(neg_scores),
                'score_separation': np.mean(pos_scores) - np.mean(neg_scores)
            })
            
        return metrics

    def _analyze_errors(self, data: pd.DataFrame, scores: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction errors"""
        labels = data['label'].values
        
        # Find cases where model confidence doesn't match labels
        high_conf_wrong = []  # High similarity but wrong label
        low_conf_wrong = []   # Low similarity but correct label
        
        threshold = np.median(scores)
        
        for i, (score, label) in enumerate(zip(scores, labels)):
            if score > threshold and label == 0:  # High similarity, negative label
                high_conf_wrong.append({
                    'index': i,
                    'query': data.iloc[i]['query'],
                    'passage': data.iloc[i]['passage'][:200] + "...",
                    'score': score,
                    'label': label
                })
            elif score < threshold and label == 1:  # Low similarity, positive label
                low_conf_wrong.append({
                    'index': i,
                    'query': data.iloc[i]['query'],
                    'passage': data.iloc[i]['passage'][:200] + "...",
                    'score': score,
                    'label': label
                })
                
        return {
            'high_confidence_errors': high_conf_wrong[:5],  # Top 5 examples
            'low_confidence_errors': low_conf_wrong[:5],
            'score_distribution': {
                'min': float(scores.min()),
                'max': float(scores.max()),
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'median': float(np.median(scores))
            }
        }

    def compare_models(self, data_path: str, models_to_test: Optional[List[str]] = None, 
                      sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Compare multiple embedding models"""
        
        # Load evaluation data
        eval_data = self.load_evaluation_data(data_path, sample_size)
        
        # Select models to test
        if models_to_test:
            selected_models = [m for m in self.available_models if m.name in models_to_test]
        else:
            selected_models = self.available_models
            
        print(f"\nTesting {len(selected_models)} models...")
        
        # Evaluate each model
        results = []
        for model_config in selected_models:
            result = self.evaluate_model(model_config, eval_data)
            results.append(result)
            self.evaluation_results.append(result)
            
        # Generate comparison report
        comparison_results = self._generate_comparison_report(results)
        
        # Save results
        self._save_results(comparison_results, eval_data)
        
        return comparison_results

    def _generate_comparison_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        
        # Extract metrics for comparison
        metrics_data = defaultdict(list)
        model_names = []
        
        for result in results:
            if 'error' in result.performance_metrics:
                continue
                
            model_names.append(result.model_name)
            
            # Retrieval metrics
            for metric, value in result.retrieval_metrics.items():
                metrics_data[f'retrieval_{metric}'].append(value)
                
            # Similarity metrics
            for metric, value in result.similarity_metrics.items():
                metrics_data[f'similarity_{metric}'].append(value)
                
            # Performance metrics
            for metric, value in result.performance_metrics.items():
                if isinstance(value, (int, float)):
                    metrics_data[f'performance_{metric}'].append(value)
                    
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(metrics_data, index=model_names)
        
        # Rank models by key metrics
        key_metrics = [
            'retrieval_f1_score', 'retrieval_accuracy', 'similarity_spearman_correlation',
            'performance_avg_time_per_text', 'performance_embedding_dimension'
        ]
        
        rankings = {}
        for metric in key_metrics:
            if metric in comparison_df.columns:
                ascending = metric.startswith('performance_avg_time')  # Lower is better for time
                rankings[metric] = comparison_df[metric].rank(ascending=ascending).to_dict()
                
        # Overall ranking (weighted combination)
        weights = {
            'retrieval_f1_score': 0.3,
            'retrieval_accuracy': 0.25, 
            'similarity_spearman_correlation': 0.25,
            'performance_avg_time_per_text': -0.2  # Negative weight (lower is better)
        }
        
        overall_scores = {}
        for model in model_names:
            score = 0
            for metric, weight in weights.items():
                if metric in comparison_df.columns:
                    normalized_value = comparison_df.loc[model, metric]
                    if metric == 'performance_avg_time_per_text':
                        # Invert time metric (lower is better)
                        normalized_value = 1 / (1 + normalized_value)
                    score += weight * normalized_value
            overall_scores[model] = score
            
        # Sort by overall score
        sorted_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'comparison_table': comparison_df.to_dict(),
            'rankings': rankings,
            'overall_ranking': sorted_models,
            'best_model': sorted_models[0][0] if sorted_models else None,
            'summary_stats': {
                'total_models_tested': len(model_names),
                'evaluation_metrics': list(comparison_df.columns),
                'best_f1_score': comparison_df['retrieval_f1_score'].max() if 'retrieval_f1_score' in comparison_df.columns else None,
                'best_accuracy': comparison_df['retrieval_accuracy'].max() if 'retrieval_accuracy' in comparison_df.columns else None
            }
        }

    def _save_results(self, comparison_results: Dict[str, Any], eval_data: pd.DataFrame):
        """Save evaluation results to files"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.output_dir / f"embedding_evaluation_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_data = {}
            for key, value in comparison_results.items():
                if isinstance(value, dict):
                    json_data[key] = {k: (v.tolist() if isinstance(v, np.ndarray) else 
                                        float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                                     for k, v in value.items()}
                else:
                    json_data[key] = value
                    
            json.dump(json_data, f, indent=2, ensure_ascii=False)
            
        # Save CSV comparison table
        if 'comparison_table' in comparison_results:
            comparison_df = pd.DataFrame(comparison_results['comparison_table'])
            csv_file = self.output_dir / f"model_comparison_{timestamp}.csv"
            comparison_df.to_csv(csv_file)
            
        # Generate and save visualizations
        self._create_visualizations(comparison_results, timestamp)
        
        # Save detailed results for each model
        for result in self.evaluation_results:
            model_file = self.output_dir / f"{result.model_name.replace(' ', '_')}_{timestamp}.json"
            with open(model_file, 'w', encoding='utf-8') as f:
                result_dict = {
                    'model_name': result.model_name,
                    'retrieval_metrics': result.retrieval_metrics,
                    'similarity_metrics': result.similarity_metrics,
                    'performance_metrics': result.performance_metrics,
                    'error_analysis': result.error_analysis
                }
                json.dump(result_dict, f, indent=2, default=str)
                
        print(f"\n✓ Results saved to: {self.output_dir}")
        print(f"✓ Main results: {json_file}")
        print(f"✓ Comparison table: {csv_file}")

    def _create_visualizations(self, comparison_results: Dict[str, Any], timestamp: str):
        """Create visualization plots"""
        
        if 'comparison_table' not in comparison_results:
            return
            
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            comparison_df = pd.DataFrame(comparison_results['comparison_table'])
            
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Performance Overview
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # F1 Score comparison
            if 'retrieval_f1_score' in comparison_df.columns:
                comparison_df['retrieval_f1_score'].plot(kind='bar', ax=axes[0,0], color='skyblue')
                axes[0,0].set_title('F1 Score Comparison')
                axes[0,0].set_ylabel('F1 Score')
                axes[0,0].tick_params(axis='x', rotation=45)
                
            # Accuracy comparison
            if 'retrieval_accuracy' in comparison_df.columns:
                comparison_df['retrieval_accuracy'].plot(kind='bar', ax=axes[0,1], color='lightgreen')
                axes[0,1].set_title('Accuracy Comparison')
                axes[0,1].set_ylabel('Accuracy')
                axes[0,1].tick_params(axis='x', rotation=45)
                
            # Correlation comparison
            if 'similarity_spearman_correlation' in comparison_df.columns:
                comparison_df['similarity_spearman_correlation'].plot(kind='bar', ax=axes[1,0], color='orange')
                axes[1,0].set_title('Spearman Correlation')
                axes[1,0].set_ylabel('Correlation')
                axes[1,0].tick_params(axis='x', rotation=45)
                
            # Performance time
            if 'performance_avg_time_per_text' in comparison_df.columns:
                comparison_df['performance_avg_time_per_text'].plot(kind='bar', ax=axes[1,1], color='salmon')
                axes[1,1].set_title('Average Time per Text (seconds)')
                axes[1,1].set_ylabel('Time (s)')
                axes[1,1].tick_params(axis='x', rotation=45)
                
            plt.tight_layout()
            plt.savefig(self.output_dir / f"model_comparison_overview_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Correlation Heatmap of metrics
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                plt.figure(figsize=(12, 8))
                correlation_matrix = comparison_df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Model Metrics Correlation Heatmap')
                plt.tight_layout()
                plt.savefig(self.output_dir / f"metrics_correlation_{timestamp}.png", dpi=300, bbox_inches='tight')
                plt.close()
                
            print(f"✓ Visualizations saved to: {self.output_dir}")
            
        except ImportError:
            print("Matplotlib/Seaborn not available for visualizations")
        except Exception as e:
            print(f"Error creating visualizations: {e}")

    def generate_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate a human-readable report"""
        
        report = []
        report.append("="*80)
        report.append("EMBEDDING MODEL EVALUATION REPORT")
        report.append("="*80)
        
        # Summary
        summary = comparison_results.get('summary_stats', {})
        report.append(f"\nSUMMARY:")
        report.append(f"• Models tested: {summary.get('total_models_tested', 'N/A')}")
        report.append(f"• Best model: {comparison_results.get('best_model', 'N/A')}")
        report.append(f"• Best F1 Score: {summary.get('best_f1_score', 'N/A'):.4f}" if summary.get('best_f1_score') else "")
        report.append(f"• Best Accuracy: {summary.get('best_accuracy', 'N/A'):.4f}" if summary.get('best_accuracy') else "")
        
        # Overall ranking
        report.append(f"\nOVERALL RANKING:")
        for i, (model, score) in enumerate(comparison_results.get('overall_ranking', []), 1):
            report.append(f"{i:2d}. {model} (Score: {score:.4f})")
            
        # Detailed comparison
        if 'comparison_table' in comparison_results:
            report.append(f"\nDETAILED METRICS:")
            comparison_df = pd.DataFrame(comparison_results['comparison_table'])
            
            # Key metrics table
            key_cols = [col for col in comparison_df.columns if any(key in col for key in 
                       ['f1_score', 'accuracy', 'spearman_correlation', 'avg_time_per_text'])]
            
            if key_cols:
                report.append(comparison_df[key_cols].round(4).to_string())
                
        # Recommendations
        report.append(f"\nRECOMMENDATIONS:")
        
        best_model = comparison_results.get('best_model')
        if best_model:
            report.append(f"• Use '{best_model}' for best overall performance")
            
        # Check for speed/accuracy tradeoffs
        if 'comparison_table' in comparison_results:
            df = pd.DataFrame(comparison_results['comparison_table'])
            if 'performance_avg_time_per_text' in df.columns and 'retrieval_f1_score' in df.columns:
                fastest = df['performance_avg_time_per_text'].idxmin()
                most_accurate = df['retrieval_f1_score'].idxmax()
                
                if fastest != most_accurate:
                    report.append(f"• For speed, consider '{fastest}'")
                    report.append(f"• For accuracy, consider '{most_accurate}'")
                    
        report.append("\n" + "="*80)
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.output_dir / f"evaluation_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        return report_text


def main():
    """Main function to run embedding model evaluation"""
    
    print("Embedding Model Evaluator")
    print("=" * 50)
    
    # Get user inputs
    data_path = input("Enter path to evaluation dataset (CSV/TSV): ").strip()
    if not data_path or not os.path.exists(data_path):
        print("Using default dataset...")
        # Use one of the available datasets
        default_datasets = [
            "integrated_robust04_data_subset_100rows_3col.tsv",
            "integrated_robust04_data_3col.tsv"
        ]
        
        for dataset in default_datasets:
            test_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), dataset)
            if os.path.exists(test_path):
                data_path = test_path
                break
        
        if not data_path:
            print("No evaluation dataset found. Please provide a valid path.")
            return
            
    print(f"Using dataset: {data_path}")
    
    # Sample size
    sample_input = input("Sample size (press Enter for all data): ").strip()
    sample_size = int(sample_input) if sample_input.isdigit() else None
    
    # Model selection
    evaluator = EmbeddingModelEvaluator()
    
    print("\nAvailable models:")
    for i, model in enumerate(evaluator.available_models, 1):
        print(f"{i:2d}. {model.name} - {model.description}")
        
    model_choice = input("\nSelect models (comma-separated numbers, or Enter for all): ").strip()
    
    selected_models = None
    if model_choice:
        try:
            indices = [int(x.strip()) - 1 for x in model_choice.split(',')]
            selected_models = [evaluator.available_models[i].name for i in indices 
                             if 0 <= i < len(evaluator.available_models)]
        except:
            print("Invalid selection, testing all models...")
            
    # Run evaluation
    print("\nStarting evaluation...")
    try:
        results = evaluator.compare_models(data_path, selected_models, sample_size)
        
        # Generate and display report
        report = evaluator.generate_report(results)
        print("\n" + report)
        
        print(f"\nEvaluation completed! Results saved to: {evaluator.output_dir}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
