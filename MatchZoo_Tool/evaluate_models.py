import matchzoo as mz
import torch
import os
import pandas as pd
from typing import Dict, List, Optional
import numpy as np
import argparse
import sys

""" 
CLI for Cross-Validation 
python evaluate_models.py --data-pack F:\SematicSearch\[method]semantic_splitter --use-cv-folds 
CLI for multiple models
python evaluate_models.py -d F:\SematicSearch\[method]semantic_splitter --use-cv-folds --models Arc-II MatchLSTM ESIM
"""

class ModelEvaluator:
    """Lớp để đánh giá và so sánh các mô hình MatchZoo"""
    
    def __init__(self, data_pack_dir: str, batch_size: int = 32, device: str = "cpu", use_cv_folds: bool = False):
        """
        Khởi tạo evaluator
        
        Args:
            data_pack_dir: Đường dẫn đến thư mục chứa test_pack.dam hoặc cv_folds/
            batch_size: Batch size cho evaluation
            device: Device để chạy evaluation ('cpu' hoặc 'cuda')
            use_cv_folds: True nếu sử dụng Cross-Validation folds
        """
        self.data_pack_dir = data_pack_dir
        self.batch_size = batch_size
        self.use_cv_folds = use_cv_folds
        
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
            print(f"Using CPU device")
        
        self.test_pack_raw = None
        self.cv_folds_data = {}  # Store CV fold data
        self.results = {}
        
        # Tải test pack hoặc CV folds
        if self.use_cv_folds:
            self._load_cv_folds()
        else:
            self._load_test_data()
    
    def _load_test_data(self):
        """Tải dữ liệu test từ test_pack.dam"""
        test_pack_path = os.path.join(self.data_pack_dir, 'test_pack.dam')
        if not os.path.exists(test_pack_path):
            raise FileNotFoundError(f"Không tìm thấy test_pack.dam trong {self.data_pack_dir}")
        
        self.test_pack_raw = mz.load_data_pack(test_pack_path)
        print(f"Đã tải test data: {len(self.test_pack_raw)} samples")
    
    def _load_cv_folds(self):
        """Tải dữ liệu từ CV folds"""
        cv_folds_dir = os.path.join(self.data_pack_dir, 'cv_folds')
        if not os.path.exists(cv_folds_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục cv_folds trong {self.data_pack_dir}")
        
        # Tìm tất cả các fold test files
        fold_files = []
        for filename in os.listdir(cv_folds_dir):
            if filename.startswith('fold_') and filename.endswith('_test.dam'):
                fold_number = int(filename.split('_')[1])
                fold_files.append((fold_number, filename))
        
        if not fold_files:
            raise FileNotFoundError(f"Không tìm thấy file fold test nào trong {cv_folds_dir}")
        
        # Sắp xếp theo số fold
        fold_files.sort(key=lambda x: x[0])
        
        # Load tất cả các folds
        for fold_num, filename in fold_files:
            fold_path = os.path.join(cv_folds_dir, filename)
            try:
                fold_data = mz.load_data_pack(fold_path)
                self.cv_folds_data[fold_num] = fold_data
                print(f"Đã tải fold {fold_num}: {len(fold_data)} samples")
            except Exception as e:
                print(f"Lỗi khi tải fold {fold_num}: {e}")
        
        print(f"Tổng cộng đã tải {len(self.cv_folds_data)} folds")
    
    def evaluate_model(self, model_dir: str, model_name: str, test_data: Optional[object] = None, fold_num: Optional[int] = None) -> Dict:
        """
        Đánh giá một mô hình trên dữ liệu test
        
        Args:
            model_dir: Đường dẫn đến thư mục chứa mô hình
            model_name: Tên mô hình
            test_data: Dữ liệu test (nếu None sẽ dùng self.test_pack_raw)
            fold_num: Số fold (dùng cho CV evaluation)
        """
        fold_suffix = f" (Fold {fold_num})" if fold_num is not None else ""
        print(f"\n--- Đánh giá mô hình: {model_name}{fold_suffix} ---")
        print(f"Device: {self.device}")
        
        # Sử dụng dữ liệu test được cung cấp hoặc mặc định
        test_pack_raw = test_data if test_data is not None else self.test_pack_raw
        
        if test_pack_raw is None:
            print("Không có dữ liệu test để đánh giá")
            return None
        
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
            test_pack_processed = preprocessor.transform(test_pack_raw)
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
        if self.use_cv_folds:
            print("BẮT ĐẦU ĐÁNH GIÁ TẤT CẢ CÁC MÔ HÌNH - CROSS VALIDATION")
        else:
            print("BẮT ĐẦU ĐÁNH GIÁ TẤT CẢ CÁC MÔ HÌNH")
        print("="*60)
        
        all_results = {}
        
        if self.use_cv_folds:
            # Cross-validation evaluation
            return self._evaluate_with_cv_folds(model_configs)
        else:
            # Single test evaluation
            for config in model_configs:
                name = config['name']
                path = config['path']
                
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
    
    def _evaluate_with_cv_folds(self, model_configs: List[Dict]) -> pd.DataFrame:
        """Đánh giá các mô hình sử dụng Cross-Validation folds"""
        all_fold_results = {}
        
        for config in model_configs:
            name = config['name']
            path = config['path']
            
            if not os.path.exists(path):
                print(f"Bỏ qua {name}: Không tìm thấy thư mục {path}")
                continue
            
            print(f"\n{'='*50}")
            print(f"Đánh giá mô hình {name} trên {len(self.cv_folds_data)} folds")
            print(f"{'='*50}")
            
            model_fold_results = []
            
            # Đánh giá trên từng fold
            for fold_num in sorted(self.cv_folds_data.keys()):
                fold_test_data = self.cv_folds_data[fold_num]
                
                print(f"\n--- Fold {fold_num} ---")
                result = self.evaluate_model(path, name, test_data=fold_test_data, fold_num=fold_num)
                
                if result:
                    # Thêm fold number vào kết quả
                    result['fold'] = fold_num
                    model_fold_results.append(result)
                else:
                    print(f"Lỗi trong fold {fold_num}")
            
            if model_fold_results:
                all_fold_results[name] = model_fold_results
                
                # Tính trung bình và std cho mô hình này
                print(f"\nKết quả tổng hợp cho {name}:")
                self._print_cv_summary(model_fold_results, name)
        
        if not all_fold_results:
            print("Không có mô hình nào được đánh giá thành công!")
            return pd.DataFrame()
        
        # Tạo DataFrame tổng hợp kết quả CV
        cv_summary_df = self._create_cv_summary_dataframe(all_fold_results)
        return cv_summary_df
    
    def _print_cv_summary(self, fold_results: List[Dict], model_name: str):
        """In tóm tắt kết quả CV cho một mô hình"""
        if not fold_results:
            return
        
        # Lấy các metrics từ fold đầu tiên
        metrics = [key for key in fold_results[0].keys() if key != 'fold']
        
        print(f"  Cross-Validation Summary cho {model_name}:")
        print(f"  {'Metric':<40} {'Mean':<10} {'Std':<10}")
        print(f"  {'-'*60}")
        
        for metric in metrics:
            values = [result[metric] for result in fold_results if metric in result]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {metric:<40} {mean_val:<10.4f} {std_val:<10.4f}")
    
    def _create_cv_summary_dataframe(self, all_fold_results: Dict) -> pd.DataFrame:
        """Tạo DataFrame tổng hợp kết quả CV"""
        summary_data = {}
        
        for model_name, fold_results in all_fold_results.items():
            if not fold_results:
                continue
            
            # Lấy các metrics
            metrics = [key for key in fold_results[0].keys() if key != 'fold']
            
            model_summary = {}
            for metric in metrics:
                values = [result[metric] for result in fold_results if metric in result]
                if values:
                    model_summary[f"{metric}_mean"] = np.mean(values)
                    model_summary[f"{metric}_std"] = np.std(values)
            
            summary_data[model_name] = model_summary
        
        if not summary_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(summary_data).T
        
        # Sắp xếp theo MAP mean
        if 'MeanAveragePrecision_mean' in df.columns:
            df = df.sort_values('MeanAveragePrecision_mean', ascending=False)
        
        return df
    
    def print_comparison_table(self, df: pd.DataFrame):
        """In bảng so sánh được định dạng đẹp với focus vào core IR metrics"""
        if df.empty:
            print("Không có dữ liệu để hiển thị!")
            return
        
        print("\n" + "="*80)
        if self.use_cv_folds:
            print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH - CROSS VALIDATION SUMMARY")
        else:
            print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH - CORE INFORMATION RETRIEVAL METRICS")
        print("="*80)
        
        if self.use_cv_folds:
            self._print_cv_comparison_table(df)
        else:
            self._print_single_test_comparison_table(df)
    
    def _print_cv_comparison_table(self, df: pd.DataFrame):
        """In bảng so sánh cho Cross-Validation results"""
        # Core IR metrics với mean và std
        core_metrics_base = [
            'MeanAveragePrecision',     # MAP
            'MeanReciprocalRank',       # MRR
            'Precision(k=1)',           # P@1
            'Precision(k=5)',           # P@5
            'Precision(k=10)',          # P@10
            'NormalizedDiscountedCumulativeGain(k=5)',   # NDCG@5
            'NormalizedDiscountedCumulativeGain(k=10)',  # NDCG@10
            'AveragePrecision'          # AP
        ]
        
        print("CROSS-VALIDATION RESULTS (Mean ± Std):")
        print("="*80)
        
        for base_metric in core_metrics_base:
            mean_col = f"{base_metric}_mean"
            std_col = f"{base_metric}_std"
            
            if mean_col in df.columns and std_col in df.columns:
                print(f"\n{base_metric}:")
                print(f"{'Model':<20} {'Mean':<12} {'Std':<12} {'Mean±Std':<20}")
                print("-" * 64)
                
                for model in df.index:
                    mean_val = df.loc[model, mean_col]
                    std_val = df.loc[model, std_col]
                    combined = f"{mean_val:.4f}±{std_val:.4f}"
                    print(f"{model:<20} {mean_val:<12.4f} {std_val:<12.4f} {combined:<20}")
        
        # Model ranking based on MAP
        if 'MeanAveragePrecision_mean' in df.columns:
            print("\n" + "="*80)
            print("MODEL RANKING (based on Mean Average Precision):")
            print("-" * 80)
            
            sorted_models = df.sort_values('MeanAveragePrecision_mean', ascending=False)
            for i, (model, row) in enumerate(sorted_models.iterrows(), 1):
                map_mean = row['MeanAveragePrecision_mean']
                map_std = row.get('MeanAveragePrecision_std', 0)
                print(f"{i:2d}. {model:<25} MAP: {map_mean:.4f} ± {map_std:.4f}")
        
        print("="*80)
    
    def _print_single_test_comparison_table(self, df: pd.DataFrame):
        """In bảng so sánh cho single test evaluation (code gốc)"""
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
        print("DCG@K METRICS:")
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
            python evaluate_models.py --data-pack F:\\SematicSearch\\[method]semantic_splitter --use-cv-folds
        """
    )
    
    parser.add_argument(
        '--data-pack', '-d',
        type=str,
        required=False,
        help='Đường dẫn đến thư mục chứa test_pack.dam hoặc cv_folds/'
    )
    
    parser.add_argument(
        '--use-cv-folds',
        action='store_true',
        help='Sử dụng Cross-Validation folds thay vì test_pack.dam'
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
    
    # Kiểm tra loại evaluation
    if args.use_cv_folds:
        cv_folds_dir = os.path.join(data_pack_dir, 'cv_folds')
        if not os.path.exists(cv_folds_dir):
            print(f"Lỗi: Không tìm thấy thư mục cv_folds trong {data_pack_dir}")
            print("Hint: Sử dụng --use-cv-folds chỉ khi có thư mục cv_folds/")
            sys.exit(1)
        print(f"Sử dụng Cross-Validation folds từ: {cv_folds_dir}")
    else:
        test_pack_path = os.path.join(data_pack_dir, 'test_pack.dam')
        if not os.path.exists(test_pack_path):
            print(f"Lỗi: Không tìm thấy test_pack.dam trong {data_pack_dir}")
            print("Hint: Sử dụng --use-cv-folds nếu bạn muốn đánh giá với CV folds")
            sys.exit(1)
    
    # Tạo evaluator với device từ CLI args
    evaluator = ModelEvaluator(data_pack_dir, batch_size=args.batch_size, device=args.device, use_cv_folds=args.use_cv_folds)
    
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
    print(f"   Evaluation mode: {'Cross-Validation' if args.use_cv_folds else 'Single Test'}")
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
