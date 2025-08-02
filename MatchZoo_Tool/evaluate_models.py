import matchzoo as mz
import torch
import os
import pandas as pd
from typing import Dict, List, Optional
import numpy as np
import argparse
import sys


class ModelEvaluator:
    """Lớp để đánh giá và so sánh các mô hình MatchZoo"""
    
    def __init__(self, data_pack_dir: str, batch_size: int = 32, device: str = "cpu"):
        """
        Khởi tạo evaluator
        
        Args:
            data_pack_dir: Đường dẫn đến thư mục chứa test_pack.dam
            batch_size: Batch size cho evaluation
            device: Device để chạy evaluation ('cpu' hoặc 'cuda')
        """
        self.data_pack_dir = data_pack_dir
        self.batch_size = batch_size
        
        # Kiểm tra và thiết lập device
        if device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("Fallback sang CPU.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            print(f"🖥️ Using CPU device")
        
        self.test_pack_raw = None
        self.results = {}
        
        # Tải test pack
        self._load_test_data()
    
    def _load_test_data(self):
        """Tải dữ liệu test"""
        test_pack_path = os.path.join(self.data_pack_dir, 'test_pack.dam')
        if not os.path.exists(test_pack_path):
            raise FileNotFoundError(f"Không tìm thấy test_pack.dam trong {self.data_pack_dir}")
        
        self.test_pack_raw = mz.load_data_pack(test_pack_path)
        print(f"Đã tải test data: {len(self.test_pack_raw)} samples")
    
    def evaluate_model(self, model_dir: str, model_name: str) -> Dict:

        print(f"\n--- Đánh giá mô hình: {model_name} ---")
        print(f"Device: {self.device}")
        
        # Kiểm tra file tồn tại
        model_path = os.path.join(model_dir, 'model.pt')
        preprocessor_path = os.path.join(model_dir, 'preprocessor')
        
        if not os.path.exists(model_path):
            print(f"Không tìm thấy model.pt trong {model_dir}")
            return None
        
        if not os.path.exists(preprocessor_path):
            print(f"Không tìm thấy preprocessor trong {model_dir}")
            return None
        
        try:
            # Tải preprocessor
            preprocessor = mz.load_preprocessor(preprocessor_path)
            print("Đã tải preprocessor")
            
            # Đọc thông tin embedding dimension từ saved model
            model_path = os.path.join(model_dir, 'model.pt')
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Lấy embedding dimension từ state_dict
            embedding_dim = state_dict['embedding.weight'].shape[1]
            vocab_size = state_dict['embedding.weight'].shape[0]
            
            # Tiền xử lý dữ liệu test
            test_pack_processed = preprocessor.transform(self.test_pack_raw)
            print("Đã tiền xử lý dữ liệu test")
            
            # Tạo test dataset
            testset = mz.dataloader.Dataset(
                data_pack=test_pack_processed,
                mode='point',
                batch_size=self.batch_size,
                shuffle=False
            )
            
            # Xác định loại mô hình để lấy đúng callback
            model_name_lower = model_name.lower().replace('-', '').replace('_', '')
            callback = self._get_model_callback(model_name_lower)

            testloader = mz.dataloader.DataLoader(
                dataset=testset,
                stage='test',
                callback=callback,
                device=self.device
            )
            print("Đã tạo test loader")
            
            # Tạo ranking task với metrics được hỗ trợ bởi MatchZoo
            ranking_task = mz.tasks.Ranking()
            ranking_task.metrics = [
                mz.metrics.MeanAveragePrecision(),  # MAP - Mean Average Precision
                mz.metrics.MeanReciprocalRank(),    # MRR - Mean Reciprocal Rank
                
                # Precision at K (P@K) 
                mz.metrics.Precision(k=1),         # P@1
                mz.metrics.Precision(k=3),         # P@3  
                mz.metrics.Precision(k=5),         # P@5
                mz.metrics.Precision(k=10),        # P@10
                mz.metrics.Precision(k=20),        # P@20
                
                # Normalized Discounted Cumulative Gain (NDCG) 
                mz.metrics.NormalizedDiscountedCumulativeGain(k=1),   # NDCG@1
                mz.metrics.NormalizedDiscountedCumulativeGain(k=3),   # NDCG@3
                mz.metrics.NormalizedDiscountedCumulativeGain(k=5),   # NDCG@5
                mz.metrics.NormalizedDiscountedCumulativeGain(k=10),  # NDCG@10
                mz.metrics.NormalizedDiscountedCumulativeGain(k=20),  # NDCG@20
                
                # Discounted Cumulative Gain (DCG) 
                mz.metrics.DiscountedCumulativeGain(k=1),    # DCG@1
                mz.metrics.DiscountedCumulativeGain(k=3),    # DCG@3
                mz.metrics.DiscountedCumulativeGain(k=5),    # DCG@5
                mz.metrics.DiscountedCumulativeGain(k=10),   # DCG@10
                mz.metrics.DiscountedCumulativeGain(k=20),   # DCG@20
                
                # Average Precision 
                mz.metrics.AveragePrecision(),
            ]
            
            # Tạo model dựa trên tên
            model_name_lower = model_name.lower().replace('-', '').replace('_', '')
            
            try:
                # Tạo model dựa trên tên
                if 'arcii' in model_name_lower or 'arc' in model_name_lower:
                    model = mz.models.ArcII()
                elif 'matchlstm' in model_name_lower:
                    model = mz.models.MatchLSTM()
                elif 'esim' in model_name_lower:
                    model = mz.models.ESIM()
                elif 'convknrm' in model_name_lower:
                    model = mz.models.ConvKNRM()
                elif 'knrm' in model_name_lower:
                    model = mz.models.KNRM()
                elif 'mvlstm' in model_name_lower:
                    model = mz.models.MVLSTM()
                elif 'pyramid' in model_name_lower or 'matchpyramid' in model_name_lower:
                    model = mz.models.MatchPyramid()
                else:
                    model = mz.models.MatchLSTM()
                
                model.params['task'] = ranking_task
                
                # Tạo và set embedding
                try:
                    if embedding_dim == 100:
                        glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=100)
                    elif embedding_dim == 200:
                        glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=200)
                    elif embedding_dim == 300:
                        glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
                    else:
                        glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=100)
                        
                    term_index = preprocessor.context['vocab_unit'].state['term_index']
                    embedding_matrix = glove_embedding.build_matrix(term_index)
                    
                    if embedding_matrix.shape[1] != embedding_dim:
                        if embedding_matrix.shape[1] < embedding_dim:
                            padding = np.zeros((embedding_matrix.shape[0], embedding_dim - embedding_matrix.shape[1]))
                            embedding_matrix = np.concatenate([embedding_matrix, padding], axis=1)
                        else:
                            embedding_matrix = embedding_matrix[:, :embedding_dim]
                    
                    l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
                    l2_norm[l2_norm == 0] = 1e-8
                    embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
                    
                    model.params['embedding'] = embedding_matrix
                    
                except Exception as e:
                    return None
                
                self._set_model_specific_params(model, model_name_lower, embedding_dim)
                
                model.build()
                model.load_state_dict(state_dict, strict=False)
                
            except Exception as e:
                return None
            
            model.to(self.device)
            model.eval()
            
            # Hiển thị thông tin GPU nếu sử dụng CUDA
            if self.device.type == "cuda":
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            
            optimizer = torch.optim.Adam(model.parameters())
            trainer = mz.trainers.Trainer(
                model=model,
                optimizer=optimizer,
                trainloader=testloader,
                validloader=testloader,
                device=self.device
            )
            
            results = trainer.evaluate(testloader)
            
            # Chuyển đổi keys từ metric objects thành strings
            results_str = {}
            for metric, value in results.items():
                if hasattr(metric, '__name__'):
                    key = metric.__name__
                elif hasattr(metric, '__class__'):
                    key = metric.__class__.__name__
                else:
                    key = str(metric)
                results_str[key] = value
            
            return results_str
            
        except Exception as e:
            print(f"Lỗi khi đánh giá mô hình {model_name}: {e}")
            return None
    
    def _set_model_specific_params(self, model, model_name_lower: str, embedding_dim: int):
        """Thiết lập các tham số đặc biệt cho từng mô hình"""
        
        def safe_set_param(param_name, value):
            if param_name in model.params:
                model.params[param_name] = value
                return True
            return False
        
        if 'arcii' in model_name_lower or 'arc' in model_name_lower:
            safe_set_param('left_length', 10)
            safe_set_param('right_length', 100)
            safe_set_param('kernel_1d_count', 32)
            safe_set_param('kernel_1d_size', 3)
            safe_set_param('kernel_2d_count', [64, 64])
            safe_set_param('kernel_2d_size', [(3, 3), (3, 3)])
            safe_set_param('pool_2d_size', [(3, 3), (3, 3)])
            safe_set_param('dropout_rate', 0.3)
            
        elif 'matchlstm' in model_name_lower:
            safe_set_param('mask_value', 0)
            safe_set_param('dropout_rate', 0.2)
            safe_set_param('dropout', 0.2)
            safe_set_param('with_matching_matrix', True)
            
        elif 'esim' in model_name_lower:
            safe_set_param('mask_value', 0)
            safe_set_param('dropout', 0.2)
            safe_set_param('dropout_rate', 0.2)
            safe_set_param('hidden_size', 200)
            safe_set_param('lstm_layer', 1)
            
        elif 'convknrm' in model_name_lower:
            safe_set_param('filters', 128)
            safe_set_param('conv_activation_func', 'tanh')
            safe_set_param('max_ngram', 3)
            safe_set_param('use_crossmatch', True)
            safe_set_param('kernel_num', 11)
            safe_set_param('sigma', 0.1)
            safe_set_param('exact_sigma', 0.001)
            
        elif 'knrm' in model_name_lower:
            safe_set_param('kernel_num', 21)
            safe_set_param('sigma', 0.1)
            safe_set_param('exact_sigma', 0.001)
            
        elif 'mvlstm' in model_name_lower:
            safe_set_param('mask_value', 0)
            safe_set_param('dropout_rate', 0.2)
            safe_set_param('dropout', 0.2)
            safe_set_param('with_match_highway', True)
            
        elif 'pyramid' in model_name_lower:
            safe_set_param('mask_value', 0)
            safe_set_param('dropout_rate', 0.2)
            safe_set_param('dropout', 0.2)
            safe_set_param('dpool_size', [3, 10])
            safe_set_param('kernel_count', [32, 32])
            safe_set_param('kernel_size', [(3, 3), (3, 3)])
            safe_set_param('activation_func', 'relu')
    
    def evaluate_all_models(self, model_configs: List[Dict]) -> pd.DataFrame:
        """
        Đánh giá tất cả các mô hình và tạo bảng so sánh
        
        Args:
            model_configs: List các dict chứa {'name': str, 'path': str}
            
        Returns:
            DataFrame chứa kết quả so sánh
        """
        print("\n" + "="*60)
        print("BẮT ĐẦU ĐÁNH GIÁ TẤT CẢ CÁC MÔ HÌNH")
        print("="*60)
        
        all_results = {}
        
        for config in model_configs:
            name = config['name']
            path =config['path']
            
            if os.path.exists(path):
                result = self.evaluate_model(path, name)
                if result:
                    all_results[name] = result
            else:
                print(f"Bỏ qua {name}: Không tìm thấy thư mục {path}")
        
        if not all_results:
            print("Không có mô hình nào được đánh giá thành công!")
            return pd.DataFrame()
        
        # Tạo DataFrame để so sánh
        df = pd.DataFrame(all_results).T
        
        # Sắp xếp theo MAP (Mean Average Precision)
        if 'MeanAveragePrecision' in df.columns:
            df = df.sort_values('MeanAveragePrecision', ascending=False)
        
        return df
    
    def print_comparison_table(self, df: pd.DataFrame):
        """In bảng so sánh được định dạng đẹp với focus vào core IR metrics"""
        if df.empty:
            print("Không có dữ liệu để hiển thị!")
            return
        
        print("\n" + "="*80)
        print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH - CORE INFORMATION RETRIEVAL METRICS")
        print("="*80)
        
        # Core IR metrics theo thứ tự ưu tiên
        core_metrics = [
            'MeanAveragePrecision',     # MAP - Most important overall metric
            'MeanReciprocalRank',       # MRR - First relevant result
            'Precision(k=1)',           # P@1 - Most strict precision  
            'Precision(k=5)',           # P@5 - Standard precision
            'Precision(k=10)',          # P@10 - Broader precision
            'NormalizedDiscountedCumulativeGain(k=5)',   # NDCG@5
            'NormalizedDiscountedCumulativeGain(k=10)',  # NDCG@10
            'DiscountedCumulativeGain(k=10)',            # DCG@10
            'AveragePrecision'          # AP - Average Precision
        ]
        
        # Lọc metrics có sẵn trong DataFrame  
        available_core_metrics = [m for m in core_metrics if m in df.columns]
        
        if available_core_metrics:
            print("CORE METRICS:")
            core_df = df[available_core_metrics]
            print(core_df.round(4))
        
        # Additional detailed metrics
        print("\n" + "-"*80)
        print("DETAILED PRECISION@K METRICS:")
        precision_metrics = [col for col in df.columns if 'Precision(k=' in col]
        if precision_metrics:
            precision_df = df[sorted(precision_metrics, key=lambda x: int(x.split('k=')[1].split(')')[0]))]
            print(precision_df.round(4))
        
        print("\n" + "-"*80)
        print("NDCG@K METRICS:")
        ndcg_metrics = [col for col in df.columns if 'NormalizedDiscountedCumulativeGain(k=' in col]
        if ndcg_metrics:
            ndcg_df = df[sorted(ndcg_metrics, key=lambda x: int(x.split('k=')[1].split(')')[0]))]
            print(ndcg_df.round(4))
        
        print("\n" + "-"*80)
        print("📊 DCG@K METRICS:")
        dcg_metrics = [col for col in df.columns if 'DiscountedCumulativeGain(k=' in col]
        if dcg_metrics:
            dcg_df = df[sorted(dcg_metrics, key=lambda x: int(x.split('k=')[1].split(')')[0]))]
            print(dcg_df.round(4))
        
        print("\n" + "="*80)
        
        # Model ranking analysis
        print("MODEL RANKING ANALYSIS:")
        if 'MeanAveragePrecision' in df.columns:
            best_map_model = df['MeanAveragePrecision'].idxmax()
            best_map_score = df.loc[best_map_model, 'MeanAveragePrecision']
            print(f"   Highest MAP: {best_map_model} ({best_map_score:.4f})")
        
        if 'MeanReciprocalRank' in df.columns:
            best_mrr_model = df['MeanReciprocalRank'].idxmax()
            best_mrr_score = df.loc[best_mrr_model, 'MeanReciprocalRank']
            print(f"   Highest MRR: {best_mrr_model} ({best_mrr_score:.4f})")
        
        if 'Precision(k=1)' in df.columns:
            best_p1_model = df['Precision(k=1)'].idxmax()
            best_p1_score = df.loc[best_p1_model, 'Precision(k=1)']
            print(f"   Highest P@1: {best_p1_model} ({best_p1_score:.4f})")
        
        if 'NormalizedDiscountedCumulativeGain(k=10)' in df.columns:
            best_ndcg10_model = df['NormalizedDiscountedCumulativeGain(k=10)'].idxmax()
            best_ndcg10_score = df.loc[best_ndcg10_model, 'NormalizedDiscountedCumulativeGain(k=10)']
            print(f"   Highest NDCG@10: {best_ndcg10_model} ({best_ndcg10_score:.4f})")
        
        # Overall performance summary
        print("\nPERFORMANCE SUMMARY:")
        if len(available_core_metrics) >= 3:
            # Calculate average rank across core metrics
            ranks = {}
            for model in df.index:
                model_ranks = []
                for metric in available_core_metrics[:5]:  # Top 5 core metrics
                    if metric in df.columns:
                        rank = df[metric].rank(ascending=False)[model]
                        model_ranks.append(rank)
                ranks[model] = np.mean(model_ranks) if model_ranks else float('inf')
            
            # Sort by average rank
            sorted_models = sorted(ranks.items(), key=lambda x: x[1])
            print("   Average Ranking (lower is better):")
            for i, (model, avg_rank) in enumerate(sorted_models, 1):
                print(f"   {i:2d}. {model:<20} (Avg Rank: {avg_rank:.2f})")
        
        print("="*80)

    
    def save_results(self, df: pd.DataFrame, filename: str = 'evaluation_results.csv'):
        """Lưu kết quả vào file CSV"""
        if not df.empty:
            df.to_csv(filename)
            print(f"Đã lưu kết quả vào {filename}")

    def _get_model_callback(self, model_name_lower: str):
        """Lấy callback phù hợp với mô hình"""
        if 'arcii' in model_name_lower or 'arc' in model_name_lower:
            return mz.models.ArcII.get_default_padding_callback(
                fixed_length_left=10,
                fixed_length_right=100,
                pad_word_value=0,
                pad_word_mode='pre'
            )
        elif 'matchlstm' in model_name_lower:
            return mz.models.MatchLSTM.get_default_padding_callback()
        elif 'esim' in model_name_lower:
            return mz.models.ESIM.get_default_padding_callback()
        elif 'convknrm' in model_name_lower:
            return mz.models.ConvKNRM.get_default_padding_callback()
        elif 'knrm' in model_name_lower:
            return mz.models.KNRM.get_default_padding_callback()
        elif 'mvlstm' in model_name_lower:
            return mz.models.MVLSTM.get_default_padding_callback()
        elif 'pyramid' in model_name_lower or 'matchpyramid' in model_name_lower:
            return mz.models.MatchPyramid.get_default_padding_callback()
        else:
            return mz.models.MatchLSTM.get_default_padding_callback()
    
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Đánh giá và so sánh các mô hình MatchZoo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python evaluate_models.py --data-pack F:\\SematicSearch\\[method]semantic_splitter
  python evaluate_models.py -d F:\\SematicSearch\\[method]semantic_splitter --output results.csv
  python evaluate_models.py --data-pack F:\\SematicSearch\\[method]semantic_splitter --models Arc-II MatchLSTM
        """
    )
    
    parser.add_argument(
        '--data-pack', '-d',
        type=str,
        required=False,
        help='Đường dẫn đến thư mục chứa test_pack.dam'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='evaluation_results.csv',
        help='Tên file output để lưu kết quả (mặc định: evaluation_results.csv)'
    )
    
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=['Arc-II'],
        choices=['Arc-II', 'MatchLSTM', 'ESIM', 'Conv-KNRM', 'KNRM', 'Match-Pyramid', 'MVLSTM'],
        help='Danh sách các mô hình cần đánh giá (mặc định: Arc-II)'
    )
    
    parser.add_argument(
        '--model-paths',
        nargs='+',
        help='Đường dẫn tương ứng cho từng mô hình (theo thứ tự với --models)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size cho evaluation (mặc định: 32)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device để chạy evaluation (mặc định: cpu)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='In thêm thông tin chi tiết'
    )
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Nhập đường dẫn đến thư mục chứa DataPack
    if hasattr(args, 'data_pack') and args.data_pack:
        data_pack_dir = args.data_pack
        print(f"Sử dụng data pack từ: {data_pack_dir}")
    else:
        data_pack_dir = input("Nhập đường dẫn đến thư mục chứa test_pack.dam: ")
    
    # Kiểm tra đường dẫn tồn tại
    if not os.path.exists(data_pack_dir):
        print(f"Lỗi: Không tìm thấy thư mục {data_pack_dir}")
        sys.exit(1)
    
    test_pack_path = os.path.join(data_pack_dir, 'test_pack.dam')
    if not os.path.exists(test_pack_path):
        print(f"Lỗi: Không tìm thấy test_pack.dam trong {data_pack_dir}")
        sys.exit(1)
    
    # Tạo evaluator với device từ CLI args
    evaluator = ModelEvaluator(data_pack_dir, batch_size=args.batch_size, device=args.device)
    
    # Cấu hình các mô hình cần đánh giá
    model_configs = []
    
    # Đường dẫn mặc định cho các mô hình
    default_model_paths = {
        'Arc-II': 'F:\\SematicSearch\\my_model\\my_model',
        'MatchLSTM': 'F:\\SematicSearch\\matchlstm_model',
        'ESIM': 'F:\\SematicSearch\\esim_model',
        'Conv-KNRM': 'F:\\SematicSearch\\conv_knrm_model',
        'KNRM': 'F:\\SematicSearch\\knrm_model',
        'Match-Pyramid': 'F:\\SematicSearch\\match_pyramid_model',
        'MVLSTM': 'F:\\SematicSearch\\mvlstm_model'
    }
    
    # Xử lý model paths
    if args.model_paths:
        if len(args.model_paths) != len(args.models):
            print(f"Lỗi: Số lượng đường dẫn mô hình ({len(args.model_paths)}) không khớp với số lượng mô hình ({len(args.models)})")
            sys.exit(1)
        
        for model_name, model_path in zip(args.models, args.model_paths):
            model_configs.append({'name': model_name, 'path': model_path})
    else:
        # Sử dụng đường dẫn mặc định
        for model_name in args.models:
            model_path = default_model_paths.get(model_name)
            if model_path:
                model_configs.append({'name': model_name, 'path': model_path})
            else:
                print(f"Không tìm thấy đường dẫn mặc định cho mô hình {model_name}")
    
    if not model_configs:
        print("Lỗi: Không có mô hình nào được cấu hình!")
        sys.exit(1)
    
    print(f"\nCấu hình đánh giá:")
    print(f"   Data pack: {data_pack_dir}")
    print(f"   Output file: {args.output}")
    print(f"   Device: {args.device}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Models: {', '.join([config['name'] for config in model_configs])}")
    
    if args.verbose:
        print(f"\nChi tiết đường dẫn mô hình:")
        for config in model_configs:
            print(f"   {config['name']}: {config['path']}")
    
    # Thực hiện đánh giá
    results_df = evaluator.evaluate_all_models(model_configs)
    
    # Hiển thị kết quả
    evaluator.print_comparison_table(results_df)
    
    # Lưu kết quả
    evaluator.save_results(results_df, args.output)

if __name__ == "__main__":
    main()
