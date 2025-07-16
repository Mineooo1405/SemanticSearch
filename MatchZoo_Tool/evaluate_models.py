import matchzoo as mz
import torch
import os
import pandas as pd
from typing import Dict, List, Optional
import numpy as np

print("=== Công cụ đánh giá mô hình MatchZoo ===")

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
        Đánh giá một mô hình cụ thể
        
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
            print(f"❌ Không tìm thấy model.pt trong {model_dir}")
            return None
        
        if not os.path.exists(preprocessor_path):
            print(f"❌ Không tìm thấy preprocessor trong {model_dir}")
            return None
        
        try:
            # Tải preprocessor
            preprocessor = mz.load_preprocessor(preprocessor_path)
            print("✅ Đã tải preprocessor")
            
            # Tiền xử lý dữ liệu test
            test_pack_processed = preprocessor.transform(self.test_pack_raw)
            print("✅ Đã tiền xử lý dữ liệu test")
            
            # Tạo test dataset
            testset = mz.dataloader.Dataset(
                data_pack=test_pack_processed,
                mode='point',
                batch_size=32,
                shuffle=False
            )
            
            # Tạo test loader
            # Xác định loại mô hình để lấy đúng callback
            if 'arcii' in model_name.lower():
                callback = mz.models.ArcII.get_default_padding_callback(
                    fixed_length_left=10,
                    fixed_length_right=100,
                    pad_word_value=0,
                    pad_word_mode='pre'
                )
            elif 'matchlstm' in model_name.lower():
                callback = mz.models.MatchLSTM.get_default_padding_callback()
            elif 'esim' in model_name.lower():
                callback = mz.models.ESIM.get_default_padding_callback()
            elif 'conv_knrm' in model_name.lower():
                callback = mz.models.ConvKNRM.get_default_padding_callback()
            elif 'knrm' in model_name.lower():
                callback = mz.models.KNRM.get_default_padding_callback()
            elif 'mvlstm' in model_name.lower():
                callback = mz.models.MVLSTM.get_default_padding_callback()
            elif 'pyramid' in model_name.lower():
                callback = mz.models.MatchPyramid.get_default_padding_callback()
            else:
                # Callback mặc định
                callback = mz.models.MatchLSTM.get_default_padding_callback()
            
            testloader = mz.dataloader.DataLoader(
                dataset=testset,
                stage='test',
                callback=callback,
                device=self.device
            )
            print("✅ Đã tạo test loader")
            
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
            
            # Tạo trainer để evaluate
            # Vì chúng ta không có model object, tạo một dummy trainer
            # Thay vào đó, ta sẽ tạo model từ state_dict
            
            # Tạo model dựa trên tên
            if 'arcii' in model_name.lower():
                model = mz.models.ArcII()
            elif 'matchlstm' in model_name.lower():
                model = mz.models.MatchLSTM()
            elif 'esim' in model_name.lower():
                model = mz.models.ESIM()
            elif 'conv_knrm' in model_name.lower():
                model = mz.models.ConvKNRM()
            elif 'knrm' in model_name.lower():
                model = mz.models.KNRM()
            elif 'mvlstm' in model_name.lower():
                model = mz.models.MVLSTM()
            elif 'pyramid' in model_name.lower():
                model = mz.models.MatchPyramid()
            else:
                raise ValueError(f"Không nhận diện được loại mô hình: {model_name}")
            
            # Thiết lập tham số cơ bản cho model
            model.params['task'] = ranking_task
            
            # Tải embedding từ preprocessor context
            try:
                # Lấy embedding từ preprocessor context nếu có
                if 'embedding' in preprocessor.context:
                    model.params['embedding'] = preprocessor.context['embedding']
                else:
                    # Tạo embedding matrix từ term_index
                    glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
                    term_index = preprocessor.context['vocab_unit'].state['term_index']
                    embedding_matrix = glove_embedding.build_matrix(term_index)
                    l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
                    l2_norm[l2_norm == 0] = 1e-8
                    embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
                    model.params['embedding'] = embedding_matrix
            except Exception as e:
                print(f"⚠️ Cảnh báo: Không thể tải embedding: {e}")
                # Tạo embedding matrix mặc định
                glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=100)
                term_index = preprocessor.context['vocab_unit'].state['term_index']
                embedding_matrix = glove_embedding.build_matrix(term_index)
                l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
                l2_norm[l2_norm == 0] = 1e-8
                embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
                model.params['embedding'] = embedding_matrix
            
            # Thiết lập các tham số đặc biệt cho từng mô hình
            if 'arcii' in model_name.lower():
                model.params['left_length'] = 10
                model.params['right_length'] = 100
                model.params['kernel_1d_count'] = 32
                model.params['kernel_1d_size'] = 3
                model.params['kernel_2d_count'] = [64, 64]
                model.params['kernel_2d_size'] = [(3, 3), (3, 3)]
                model.params['pool_2d_size'] = [(3, 3), (3, 3)]
                model.params['dropout_rate'] = 0.3
            elif 'matchlstm' in model_name.lower():
                model.params['mask_value'] = 0
            elif 'esim' in model_name.lower():
                model.params['mask_value'] = 0
                model.params['dropout'] = 0.2
                model.params['hidden_size'] = 200
                model.params['lstm_layer'] = 1
            elif 'conv_knrm' in model_name.lower():
                model.params['filters'] = 128
                model.params['conv_activation_func'] = 'tanh'
                model.params['max_ngram'] = 3
                model.params['use_crossmatch'] = True
                model.params['kernel_num'] = 11
                model.params['sigma'] = 0.1
                model.params['exact_sigma'] = 0.001
            elif 'knrm' in model_name.lower():
                model.params['kernel_num'] = 21
                model.params['sigma'] = 0.1
                model.params['exact_sigma'] = 0.001
            elif 'mvlstm' in model_name.lower():
                # MVLSTM có thể không cần tham số đặc biệt
                pass
            elif 'pyramid' in model_name.lower():
                # MatchPyramid có thể không cần tham số đặc biệt
                pass
            
            # Build model
            model.build()
            
            # Tải trọng số
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            print("✅ Đã tải model và trọng số")
            
            # Tạo trainer để evaluate
            optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer
            trainer = mz.trainers.Trainer(
                model=model,
                optimizer=optimizer,
                trainloader=testloader,  # Dummy trainloader
                validloader=testloader,
                device=self.device
            )
            
            # Thực hiện evaluation
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
            
            print("✅ Đã hoàn thành evaluation")
            print("Kết quả:")
            for metric, value in results_str.items():
                print(f"  {metric}: {value:.4f}")
            
            return results_str
            
        except Exception as e:
            print(f"❌ Lỗi khi đánh giá mô hình {model_name}: {e}")
            return None
    
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
            path ="F:/SematicSearch/trained_models/" + config['path']
            
            if os.path.exists(path):
                result = self.evaluate_model(path, name)
                if result:
                    all_results[name] = result
            else:
                print(f"⚠️ Bỏ qua {name}: Không tìm thấy thư mục {path}")
        
        if not all_results:
            print("❌ Không có mô hình nào được đánh giá thành công!")
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
            print(f"🏆 MÔ HÌNH TỐT NHẤT: {best_model} (MAP: {best_score:.4f})")
        
        print("="*80)
    
    def save_results(self, df: pd.DataFrame, filename: str = 'evaluation_results.csv'):
        """Lưu kết quả vào file CSV"""
        if not df.empty:
            df.to_csv(filename)
            print(f"💾 Đã lưu kết quả vào {filename}")

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
    
    print("\n🎉 Hoàn thành việc đánh giá!")

if __name__ == "__main__":
    main()
