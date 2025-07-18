import matchzoo as mz
import torch
import os
import pandas as pd
from typing import Dict, List, Optional
import numpy as np


class ModelEvaluator:
    """Lớp để đánh giá và so sánh các mô hình MatchZoo"""
    
    def __init__(self, data_pack_dir: str):
        """
        Khởi tạo evaluator
        
        Args:
            data_pack_dir: Đường dẫn đến thư mục chứa test_pack.dam
        """
        self.data_pack_dir = data_pack_dir
        self.device = torch.device("cpu")
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
        """
        Đánh giá một mô hình cụ thể với cải tiến xử lý embedding dimension
        
        Args:
            model_dir: Đường dẫn đến thư mục chứa model và preprocessor
            model_name: Tên mô hình để hiển thị
            
        Returns:
            Dict chứa kết quả đánh giá
        """
        print(f"\n--- Đánh giá mô hình: {model_name} ---")
        
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
                batch_size=32,
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
            
            # Tạo ranking task với metrics
            ranking_task = mz.tasks.Ranking()
            ranking_task.metrics = [
                mz.metrics.NormalizedDiscountedCumulativeGain(k=1),
                mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
                mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
                mz.metrics.NormalizedDiscountedCumulativeGain(k=10),
                mz.metrics.MeanAveragePrecision(),
                mz.metrics.Precision(k=1),
                mz.metrics.Precision(k=3),
                mz.metrics.Precision(k=5),
                mz.metrics.Precision(k=10)
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
            path ="F:/SematicSearch/Trained_model/" + config['path']
            
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
        """In bảng so sánh được định dạng đẹp"""
        if df.empty:
            print("Không có dữ liệu để hiển thị!")
            return
        
        print("\n" + "="*80)
        print("BẢNG SO SÁNH KẾT QUẢ CÁC MÔ HÌNH")
        print("="*80)
        
        # Chọn các metrics quan trọng nhất để hiển thị
        important_metrics = [
            'MeanAveragePrecision',
            'NormalizedDiscountedCumulativeGain(k=1)',
            'NormalizedDiscountedCumulativeGain(k=3)',
            'NormalizedDiscountedCumulativeGain(k=5)',
            'NormalizedDiscountedCumulativeGain(k=10)',
            'Precision(k=1)',
            'Precision(k=3)',
            'Precision(k=5)'
        ]
        
        # Lọc các metrics có trong DataFrame
        available_metrics = [m for m in important_metrics if m in df.columns]
        
        if available_metrics:
            display_df = df[available_metrics]
            print(display_df.round(4))
        else:
            print(df.round(4))
        
        print("\n" + "="*80)
        
        # Tìm mô hình tốt nhất
        if 'MeanAveragePrecision' in df.columns:
            best_model = df['MeanAveragePrecision'].idxmax()
            best_score = df.loc[best_model, 'MeanAveragePrecision']
            print(f"MÔ HÌNH TỐT NHẤT: {best_model} (MAP: {best_score:.4f})")
        
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
    
def main():
    # Nhập đường dẫn đến thư mục chứa DataPack
    data_pack_dir = input("Nhập đường dẫn đến thư mục chứa test_pack.dam: ")
    
    # Tạo evaluator
    evaluator = ModelEvaluator(data_pack_dir)
    
    # Cấu hình các mô hình cần đánh giá
    model_configs = [
        {'name': 'Arc-II', 'path': 'arcii_model'},
        {'name': 'MatchLSTM', 'path': 'matchlstm_model'},
        {'name': 'ESIM', 'path': 'esim_model'},
        {'name': 'Conv-KNRM', 'path': 'conv_knrm_model'},
        {'name': 'KNRM', 'path': 'knrm_model'},
        {'name': 'Match-Pyramid', 'path': 'match_pyramid_model'},
        {'name': 'MVLSTM', 'path': 'mvlstm_model'}
    ]
    
    # Thực hiện đánh giá
    results_df = evaluator.evaluate_all_models(model_configs)
    
    # Hiển thị kết quả
    evaluator.print_comparison_table(results_df)
    
    # Lưu kết quả
    evaluator.save_results(results_df)
    
    print("\nHoàn thành đánh giá!")

if __name__ == "__main__":
    main()
