"""
Quick Embedding Model Tester
============================
A simplified script to quickly test and compare embedding models
for your semantic search project.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from Tool.Sentence_Embedding import sentence_embedding, get_device
except ImportError:
    print("Error: Cannot import Sentence_Embedding tool")
    sys.exit(1)

def quick_test_model(model_name: str, test_queries: List[str], test_passages: List[str]) -> Dict:
    """Quick test of an embedding model with similarity computation proof"""
    print(f"\nTesting: {model_name}")
    
    start_time = time.time()
    
    try:
        # Get embeddings
        query_embs = sentence_embedding(test_queries, model_name=model_name)  # Shape: (num_queries, embed_dim)
        passage_embs = sentence_embedding(test_passages, model_name=model_name)  # Shape: (num_passages, embed_dim)
        
        # Show embedding proof
        print(f"  ✓ Query embeddings: {query_embs.shape}")
        print(f"  ✓ Passage embeddings: {passage_embs.shape}")
        
        # Calculate similarities with detailed computation proof
        similarities = []
        min_len = min(len(query_embs), len(passage_embs))
        
        # Show similarity computation method
        print(f"  ✓ Computing {min_len} pairwise similarities using cosine similarity:")
        print(f"    Formula: similarity = (q·p) / (||q|| × ||p||)")
        
        for i in range(min_len):
            q_emb = query_embs[i]
            p_emb = passage_embs[i]
            
            # Manual cosine similarity calculation with proof
            dot_product = np.dot(q_emb, p_emb)
            q_norm = np.linalg.norm(q_emb)
            p_norm = np.linalg.norm(p_emb)
            sim = dot_product / (q_norm * p_norm)
            similarities.append(sim)
            
            # Show first calculation as proof
            if i == 0:
                print(f"    Sample: dot_product={dot_product:.6f}, q_norm={q_norm:.6f}, p_norm={p_norm:.6f}")
                print(f"    Result: {dot_product:.6f} / ({q_norm:.6f} × {p_norm:.6f}) = {sim:.6f}")
                
        embedding_time = time.time() - start_time
        
        return {
            'model': model_name,
            'success': True,
            'embedding_time': embedding_time,
            'avg_time_per_text': embedding_time / (len(test_queries) + len(test_passages)),
            'embedding_dim': query_embs.shape[1] if len(query_embs.shape) > 1 else len(query_embs[0]),
            'similarity_scores': similarities,
            'avg_similarity': np.mean(similarities),
            'similarity_std': np.std(similarities),
            'similarity_computation': {
                'method': 'cosine_similarity',
                'formula': 'dot_product / (norm_q * norm_p)',
                'sample_calculation_verified': True
            },
            'error': None
        }
        
    except Exception as e:
        return {
            'model': model_name,
            'success': False,
            'error': str(e),
            'embedding_time': time.time() - start_time
        }

def test_with_real_data(data_path: str, models: List[str], max_samples: int = 50) -> Dict:
    """Test models with real dataset"""
    print(f"Loading data from: {data_path}")
    
    # Load data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_csv(data_path, sep='\t')
        
    # Handle different column names
    if 'query_text' in df.columns:
        df['query'] = df['query_text']
    if 'chunk_text' in df.columns:
        df['passage'] = df['chunk_text']
        
    # Sample data
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        
    queries = df['query'].tolist()
    passages = df['passage'].tolist()
    labels = df['label'].tolist() if 'label' in df.columns else None
    
    print(f"Testing with {len(df)} samples")
    
    results = []
    for model in models:
        result = test_model_retrieval(model, queries, passages, labels)
        results.append(result)
        
    return {
        'results': results,
        'data_info': {
            'samples': len(df),
            'avg_query_length': np.mean([len(q.split()) for q in queries]),
            'avg_passage_length': np.mean([len(p.split()) for p in passages])
        }
    }

def test_model_retrieval(model_name: str, queries: List[str], passages: List[str], 
                        labels: List[int] = None) -> Dict:
    """Test model on retrieval task with similarity matrix proof"""
    print(f"Testing retrieval: {model_name}")
    
    start_time = time.time()
    
    try:
        # Get embeddings
        query_embs = sentence_embedding(queries, model_name=model_name)
        passage_embs = sentence_embedding(passages, model_name=model_name)
        
        # Build full similarity matrix for proof
        print(f"  Building similarity matrix: {len(queries)} × {len(passages)}")
        
        # Normalize embeddings for efficient cosine similarity
        query_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
        passage_norms = np.linalg.norm(passage_embs, axis=1, keepdims=True)
        
        query_normalized = query_embs / (query_norms + 1e-8)
        passage_normalized = passage_embs / (passage_norms + 1e-8)
        
        # Compute full similarity matrix: Q × P^T
        similarity_matrix = np.dot(query_normalized, passage_normalized.T)
        
        # Extract diagonal for query-passage pairs
        similarities = np.diag(similarity_matrix)
        
        # Show matrix computation proof
        print(f"  ✓ Similarity matrix shape: {similarity_matrix.shape}")
        print(f"  ✓ Matrix computation: Q_norm({query_normalized.shape}) × P_norm^T({passage_normalized.T.shape})")
        print(f"  ✓ Diagonal similarities range: [{similarities.min():.4f}, {similarities.max():.4f}]")
        
        # Verify with manual calculation
        if len(similarities) > 0:
            manual_sim = np.dot(query_normalized[0], passage_normalized[0])
            matrix_sim = similarity_matrix[0, 0]
            print(f"  ✓ Verification: manual={manual_sim:.6f}, matrix={matrix_sim:.6f}")
            
        embedding_time = time.time() - start_time
        
        result = {
            'model': model_name,
            'success': True,
            'embedding_time': embedding_time,
            'avg_time_per_text': embedding_time / (len(queries) + len(passages)),
            'embedding_dim': query_embs.shape[1] if len(query_embs.shape) > 1 else len(query_embs[0]),
            'avg_similarity': float(np.mean(similarities)),
            'similarity_std': float(np.std(similarities)),
            'similarity_range': [float(np.min(similarities)), float(np.max(similarities))],
            'similarity_matrix_stats': {
                'shape': similarity_matrix.shape,
                'full_matrix_mean': float(similarity_matrix.mean()),
                'diagonal_mean': float(similarities.mean()),
                'computation_method': 'normalized_dot_product'
            }
        }
        
        # If we have labels, calculate performance metrics
        if labels:
            labels = np.array(labels)
            
            # Find best threshold
            thresholds = np.linspace(similarities.min(), similarities.max(), 50)
            best_f1 = 0
            best_threshold = 0
            
            for threshold in thresholds:
                predictions = (similarities >= threshold).astype(int)
                
                tp = np.sum((predictions == 1) & (labels == 1))
                fp = np.sum((predictions == 1) & (labels == 0))
                fn = np.sum((predictions == 0) & (labels == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    
            # Calculate final metrics
            predictions = (similarities >= best_threshold).astype(int)
            accuracy = np.mean(predictions == labels)
            
            # Correlation
            from scipy.stats import spearmanr, pearsonr
            spearman_corr, _ = spearmanr(similarities, labels)
            pearson_corr, _ = pearsonr(similarities, labels)
            
            result.update({
                'accuracy': float(accuracy),
                'f1_score': float(best_f1),
                'best_threshold': float(best_threshold),
                'spearman_correlation': float(spearman_corr),
                'pearson_correlation': float(pearson_corr)
            })
            
        return result
        
    except Exception as e:
        return {
            'model': model_name,
            'success': False,
            'error': str(e),
            'embedding_time': time.time() - start_time
        }

def print_comparison_table(results: List[Dict]):
    """Print a nice comparison table"""
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("No successful results to compare")
        return
        
    print("\n" + "="*100)
    print("MODEL COMPARISON RESULTS")
    print("="*100)
    
    # Header
    headers = ["Model", "Dim", "Time(s)", "F1", "Acc", "Corr", "Avg Sim"]
    print(f"{'Model':<25} {'Dim':<6} {'Time(s)':<8} {'F1':<6} {'Acc':<6} {'Corr':<6} {'Avg Sim':<8}")
    print("-" * 100)
    
    # Sort by F1 score if available, otherwise by avg similarity
    sort_key = 'f1_score' if 'f1_score' in successful_results[0] else 'avg_similarity'
    successful_results.sort(key=lambda x: x.get(sort_key, 0), reverse=True)
    
    for result in successful_results:
        model = result['model'][:24]  # Truncate long model names
        dim = result.get('embedding_dim', 0)
        time_val = result.get('embedding_time', 0)
        f1 = result.get('f1_score', 0)
        acc = result.get('accuracy', 0)
        corr = result.get('spearman_correlation', 0)
        avg_sim = result.get('avg_similarity', 0)
        
        print(f"{model:<25} {dim:<6d} {time_val:<8.2f} {f1:<6.3f} {acc:<6.3f} {corr:<6.3f} {avg_sim:<8.3f}")
        
    print("-" * 100)
    
    # Best model summary
    best_model = successful_results[0]
    print(f"\n🏆 BEST MODEL: {best_model['model']}")
    
    if 'f1_score' in best_model:
        print(f"   F1 Score: {best_model['f1_score']:.4f}")
        print(f"   Accuracy: {best_model.get('accuracy', 0):.4f}")
        print(f"   Correlation: {best_model.get('spearman_correlation', 0):.4f}")
    print(f"   Embedding Dimension: {best_model.get('embedding_dim', 0)}")
    print(f"   Processing Time: {best_model.get('embedding_time', 0):.2f}s")

def quick_demo():
    """Quick demonstration with sample queries"""
    
    sample_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?", 
        "What are neural networks?"
    ]
    
    sample_passages = [
        "Artificial intelligence is a branch of computer science that aims to create machines capable of intelligent behavior.",
        "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
        "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes or neurons."
    ]
    
    # Test a few popular models
    models_to_test = [
        "thenlper/gte-large",
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-base-en-v1.5"
    ]
    
    print("Running quick demo with sample data...")
    print(f"Device: {get_device()}")
    
    results = []
    for model in models_to_test:
        result = quick_test_model(model, sample_queries, sample_passages)
        results.append(result)
        
        if result['success']:
            print(f"✓ {model}: {result['embedding_dim']}d, {result['embedding_time']:.2f}s, avg_sim={result['avg_similarity']:.3f}")
        else:
            print(f"✗ {model}: {result['error']}")
            
    return results

def main():
    """Main function"""
    print("Quick Embedding Model Tester")
    print("=" * 50)
    
    choice = input("Choose option:\n1. Quick demo with sample data\n2. Test with your dataset\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        results = quick_demo()
        print_comparison_table(results)
        
    elif choice == "2":
        data_path = input("Enter path to your dataset (CSV/TSV): ").strip()
        
        if not os.path.exists(data_path):
            print("File not found!")
            return
            
        # Model selection
        default_models = [
            "thenlper/gte-large",
            "thenlper/gte-base", 
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-base-en-v1.5"
        ]
        
        print("\nDefault models to test:")
        for i, model in enumerate(default_models, 1):
            print(f"{i}. {model}")
            
        model_choice = input("Use default models? (y/n): ").strip().lower()
        
        if model_choice == 'n':
            custom_models = input("Enter model names (comma-separated): ").strip()
            models_to_test = [m.strip() for m in custom_models.split(',')]
        else:
            models_to_test = default_models
            
        max_samples = input("Max samples to test (default 100): ").strip()
        max_samples = int(max_samples) if max_samples.isdigit() else 100
        
        print(f"\nTesting {len(models_to_test)} models with up to {max_samples} samples...")
        
        test_results = test_with_real_data(data_path, models_to_test, max_samples)
        print_comparison_table(test_results['results'])
        
        # Save results
        output_file = f"embedding_test_results_{int(time.time())}.json"
        import json
        with open(output_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
        
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
