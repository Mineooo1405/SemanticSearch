# -*- coding: utf-8 -*-
"""
MatchZoo Training Controller with CLI
====================================
CLI interface để train các mô hình MatchZoo với dataset đã được chunked.

Usage Examples:
    # Train một model cụ thể
    python train_controller.py --model ArcII --data-pack f:/SematicSearch/[method]semantic_grouping/train_pack.dam --epochs 10 --device cpu

    # Train nhiều models liên tiếp
    python train_controller.py --model ArcII KNRM ESIM --data-pack f:/SematicSearch/[method]semantic_grouping/train_pack.dam --epochs 5

    # Sử dụng custom output directory
    python train_controller.py --model ArcII --data-pack f:/path/to/datapack --output-dir ./custom_models --epochs 10

    # Với device cụ thể
    python train_controller.py --model ArcII --data-pack f:/path/to/datapack --device cuda --batch-size 32

Features:
    - Hỗ trợ tất cả các model: ArcII, KNRM, Conv-KNRM, ESIM, MatchLSTM, MatchPyramid, MVLSTM
    - Device auto-detection: DirectML (AMD GPU), CUDA (NVIDIA), CPU fallback
    - Configurable hyperparameters qua CLI
    - Progress logging và model saving
    - Error handling và validation
"""

import argparse
import os
import sys
import gc
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matchzoo as mz

# ===== Model Configurations =====
MODEL_CONFIGS = {
    'ArcII': {
        'class': mz.models.ArcII,
        'preprocessor': lambda: mz.models.ArcII.get_default_preprocessor(
            filter_mode='df',
            filter_low_freq=2,
        ),
        'padding_callback': lambda: mz.models.ArcII.get_default_padding_callback(
            fixed_length_left=10,
            fixed_length_right=256,
            pad_word_value=0,
            pad_word_mode='pre'
        ),
        'model_params': {
            'left_length': 10,
            'right_length': 256,
            'kernel_1d_count': 32,
            'kernel_1d_size': 3,
            'kernel_2d_count': [64, 64],
            'kernel_2d_size': [(3, 3), (3, 3)],
            'pool_2d_size': [(3, 3), (3, 3)],
            'dropout_rate': 0.3
        },
        'training_params': {
            'optimizer_class': torch.optim.Adam,
            'batch_size': 20,
            'num_dup': 2,
            'num_neg': 1,
        }
    },
    'KNRM': {
        'class': mz.models.KNRM,
        'preprocessor': lambda: mz.preprocessors.BasicPreprocessor(
            truncated_length_left=10,
            truncated_length_right=40,
            filter_low_freq=2
        ),
        'padding_callback': lambda: mz.models.KNRM.get_default_padding_callback(),
        'model_params': {
            'kernel_num': 21,
            'sigma': 0.1,
            'exact_sigma': 0.001
        },
        'training_params': {
            'optimizer_class': torch.optim.Adadelta,
            'batch_size': 20,
            'num_dup': 5,
            'num_neg': 1,
        }
    },
    'Conv-KNRM': {
        'class': mz.models.ConvKNRM,
        'preprocessor': lambda: mz.models.ConvKNRM.get_default_preprocessor(),
        'padding_callback': lambda: mz.models.ConvKNRM.get_default_padding_callback(),
        'model_params': {
            'filters': 128,
            'conv_activation_func': 'tanh',
            'max_ngram': 3,
            'use_crossmatch': True,
            'kernel_num': 11,
            'sigma': 0.1,
            'exact_sigma': 0.001
        },
        'training_params': {
            'optimizer_class': torch.optim.Adadelta,
            'optimizer_kwargs': {},
            'scheduler_class': torch.optim.lr_scheduler.StepLR,
            'scheduler_kwargs': {'step_size': 3},
            'batch_size': 20,
            'num_dup': 5,
            'num_neg': 1,
            'clip_norm': 10
        }
    },
    'ESIM': {
        'class': mz.models.ESIM,
        'preprocessor': lambda: mz.models.ESIM.get_default_preprocessor(),
        'padding_callback': lambda: mz.models.ESIM.get_default_padding_callback(),
        'model_params': {
            'mask_value': 0,
            'dropout': 0.2,
            'hidden_size': 200,
            'lstm_layer': 1
        },
        'training_params': {
            'optimizer_class': torch.optim.Adadelta,
            'batch_size': 20,
            'num_dup': 5,
            'num_neg': 10,
            'loss': mz.losses.RankCrossEntropyLoss(num_neg=10)
        }
    },
    'MatchLSTM': {
        'class': mz.models.MatchLSTM,
        'preprocessor': lambda: mz.models.MatchLSTM.get_default_preprocessor(),
        'padding_callback': lambda: mz.models.MatchLSTM.get_default_padding_callback(),
        'model_params': {
            'mask_value': 0
        },
        'training_params': {
            'optimizer_class': torch.optim.Adadelta,
            'batch_size': 20,
            'num_dup': 5,
            'num_neg': 10,
            'loss': mz.losses.RankCrossEntropyLoss(num_neg=10)
        }
    },
    'MatchPyramid': {
        'class': mz.models.MatchPyramid,
        'preprocessor': lambda: mz.models.MatchPyramid.get_default_preprocessor(),
        'padding_callback': lambda: mz.models.MatchPyramid.get_default_padding_callback(),
        'model_params': {},
        'training_params': {
            'optimizer_class': torch.optim.Adam,
            'batch_size': 16,
            'num_dup': 5,
            'num_neg': 1,
        }
    },
    'MVLSTM': {
        'class': mz.models.MVLSTM,
        'preprocessor': lambda: mz.models.MVLSTM.get_default_preprocessor(),
        'padding_callback': lambda: mz.models.MVLSTM.get_default_padding_callback(),
        'model_params': {},
        'training_params': {
            'optimizer_class': torch.optim.Adadelta,
            'batch_size': 16,
            'num_dup': 5,
            'num_neg': 10,
            'loss': mz.losses.RankCrossEntropyLoss(num_neg=10)
        }
    }
}

# ===== Device Configuration =====
def configure_device(device_preference: str = "auto") -> torch.device:
    """
    Tự động phát hiện và cấu hình device tốt nhất.
    
    Args:
        device_preference: "auto", "cuda", "dml", hoặc "cpu"
        
    Returns:
        torch.device object
    """
    if device_preference.lower() == "auto":
        # Try DirectML first (AMD GPU)
        try:
            import torch_directml
            if torch_directml.is_available():
                device = torch_directml.device()
                print(f"Sử dụng DirectML (AMD GPU): {device}")
                return device
        except ImportError:
            pass
        
        # Try CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Sử dụng CUDA (NVIDIA GPU): {device}")
            return device
        
        # Fallback to CPU
        device = torch.device("cpu")
        print(f" Sử dụng CPU (không tìm thấy GPU khả dụng)")
        return device
    
    elif device_preference.lower() == "dml":
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"Sử dụng DirectML: {device}")
            return device
        except ImportError:
            print("DirectML không khả dụng, fallback to CPU")
            return torch.device("cpu")
    
    elif device_preference.lower() == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Sử dụng CUDA: {device}")
            return device
        else:
            print("CUDA không khả dụng, fallback to CPU")
            return torch.device("cpu")
    
    else:  # cpu
        device = torch.device("cpu")
        print(f"Sử dụng CPU")
        return device

# ===== Training Class =====
class ModelTrainer:
    """Class để huấn luyện các mô hình MatchZoo"""
    
    def __init__(self, 
                 data_pack_dir: str, 
                 output_dir: str = "Trained_model",
                 device: str = "auto",
                 verbose: bool = True):
        self.data_pack_dir = Path(data_pack_dir)
        self.output_dir = Path(output_dir)
        self.device = configure_device(device)
        self.verbose = verbose
        
        # Tạo output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Validate data pack directory
        if not self.data_pack_dir.exists():
            raise FileNotFoundError(f"Data pack directory không tồn tại: {data_pack_dir}")
        
        self._load_data_packs()
        
    def _load_data_packs(self):
        """Tải DataPacks từ thư mục"""
        train_pack_path = self.data_pack_dir / 'train_pack.dam'
        test_pack_path = self.data_pack_dir / 'test_pack.dam'
        
        if not train_pack_path.exists():
            raise FileNotFoundError(f"Không tìm thấy train_pack.dam trong {self.data_pack_dir}")
        if not test_pack_path.exists():
            raise FileNotFoundError(f"Không tìm thấy test_pack.dam trong {self.data_pack_dir}")
        
        if self.verbose:
            print(f"Loading DataPacks từ {self.data_pack_dir}")
        
        self.train_pack_raw = mz.load_data_pack(str(train_pack_path))
        self.test_pack_raw = mz.load_data_pack(str(test_pack_path))
        
        if self.verbose:
            print(f"Train: {len(self.train_pack_raw)} samples")
            print(f"Test: {len(self.test_pack_raw)} samples")
    
    def train_model(self, 
                    model_name: str,
                    epochs: int = 5,
                    batch_size: Optional[int] = None,
                    learning_rate: Optional[float] = None,
                    custom_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Train một model cụ thể.
        
        Args:
            model_name: Tên model ('ArcII', 'KNRM', etc.)
            epochs: Số epochs để train
            batch_size: Override batch size mặc định
            learning_rate: Override learning rate mặc định  
            custom_params: Override model parameters
            
        Returns:
            Đường dẫn đến model đã được lưu
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model '{model_name}' không được hỗ trợ. "
                           f"Các model khả dụng: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_name]
        model_output_dir = self.output_dir / f"{model_name.lower()}_model"
        model_output_dir.mkdir(exist_ok=True)
        
        if self.verbose:
            print(f"\n🚀 Bắt đầu huấn luyện model {model_name}")
            print(f"   📁 Output: {model_output_dir}")
            print(f"   🔄 Epochs: {epochs}")
            print(f"   💾 Device: {self.device}")
        
        try:
            # 1. Setup task và metrics
            task = self._setup_task(config)
            
            # 2. Preprocessing
            preprocessor = config['preprocessor']()
            train_pack_processed = preprocessor.fit_transform(self.train_pack_raw)
            test_pack_processed = preprocessor.transform(self.test_pack_raw)
            
            if self.verbose:
                print("Tiền xử lý dữ liệu hoàn tất")
            
            # 3. Embedding matrix
            embedding_matrix = self._create_embedding_matrix(preprocessor)
            
            # 4. DataLoaders
            trainloader, testloader = self._create_dataloaders(
                train_pack_processed, 
                test_pack_processed,
                config,
                batch_size or config['training_params']['batch_size']
            )
            
            # 5. Build model
            model = self._build_model(config, task, embedding_matrix, custom_params)
            
            # 6. Setup optimizer
            optimizer, scheduler = self._setup_optimizer(model, config, learning_rate)
            
            # 7. Train
            trainer = self._setup_trainer(
                model, optimizer, trainloader, testloader, 
                epochs, model_output_dir, config, scheduler
            )
            
            if self.verbose:
                print(f"Bắt đầu huấn luyện với {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
            
            trainer.run()
            
            # 8. Save
            trainer.save_model()
            preprocessor.save(str(model_output_dir / 'preprocessor'))
            
            if self.verbose:
                print(f"Hoàn thành! Model được lưu tại: {model_output_dir}")
            
            return str(model_output_dir)
            
        except Exception as e:
            print(f"Lỗi khi huấn luyện {model_name}: {e}")
            raise
        finally:
            # Cleanup memory
            gc.collect()
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
    
    def _setup_task(self, config: Dict[str, Any]) -> mz.tasks.Ranking:
        """Setup ranking task với metrics"""
        training_params = config['training_params']
        
        # Check for custom loss
        loss = training_params.get('loss')
        if loss:
            task = mz.tasks.Ranking(losses=loss)
        else:
            task = mz.tasks.Ranking()
        
        # Standard metrics cho IR
        task.metrics = [
            mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
            mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
            mz.metrics.MeanAveragePrecision()
        ]
        
        return task
    
    def _create_embedding_matrix(self, preprocessor) -> np.ndarray:
        """Tạo embedding matrix từ GloVe"""
        glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=100)
        term_index = preprocessor.context['vocab_unit'].state['term_index']
        embedding_matrix = glove_embedding.build_matrix(term_index)
        
        # L2 normalize
        l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
        l2_norm[l2_norm == 0] = 1e-8
        embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
        
        del glove_embedding
        gc.collect()
        
        if self.verbose:
            print(f"Embedding matrix: {embedding_matrix.shape}")
        
        return embedding_matrix
    
    def _create_dataloaders(self, 
                           train_pack_processed, 
                           test_pack_processed,
                           config: Dict[str, Any],
                           batch_size: int) -> Tuple[mz.dataloader.DataLoader, mz.dataloader.DataLoader]:
        """Tạo DataLoaders cho training và testing"""
        training_params = config['training_params']
        
        trainset = mz.dataloader.Dataset(
            data_pack=train_pack_processed,
            mode='pair',
            num_dup=training_params['num_dup'],
            num_neg=training_params['num_neg'],
            batch_size=batch_size,
            resample=True,
            sort=False,
            shuffle=True
        )
        
        testset = mz.dataloader.Dataset(
            data_pack=test_pack_processed,
            mode='point',
            batch_size=batch_size,
            shuffle=False
        )
        
        padding_callback = config['padding_callback']()
        
        trainloader = mz.dataloader.DataLoader(
            dataset=trainset,
            stage='train',
            callback=padding_callback,
            device=self.device
        )
        
        testloader = mz.dataloader.DataLoader(
            dataset=testset,
            stage='dev',
            callback=padding_callback,
            device=self.device
        )
        
        if self.verbose:
            print(f"DataLoaders tạo thành công (batch_size={batch_size})")
        
        return trainloader, testloader
    
    def _build_model(self, 
                     config: Dict[str, Any],
                     task: mz.tasks.Ranking,
                     embedding_matrix: np.ndarray,
                     custom_params: Optional[Dict[str, Any]] = None) -> mz.models.Model:
        """Build và configure model"""
        model = config['class']()
        model.params['task'] = task
        model.params['embedding'] = embedding_matrix
        
        # Apply default model parameters
        for key, value in config['model_params'].items():
            model.params[key] = value
        
        # Apply custom parameters if provided
        if custom_params:
            for key, value in custom_params.items():
                model.params[key] = value
                if self.verbose:
                    print(f"Override parameter: {key} = {value}")
        
        model.build()
        model.to(self.device)
        
        return model
    
    def _setup_optimizer(self, 
                        model: mz.models.Model,
                        config: Dict[str, Any],
                        learning_rate: Optional[float] = None) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Setup optimizer và scheduler"""
        training_params = config['training_params']
        optimizer_class = training_params['optimizer_class']
        
        # Create optimizer
        optimizer_kwargs = training_params.get('optimizer_kwargs', {})
        if learning_rate:
            optimizer_kwargs['lr'] = learning_rate
            
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        
        # Create scheduler if specified
        scheduler = None
        if 'scheduler_class' in training_params:
            scheduler_class = training_params['scheduler_class']
            scheduler_kwargs = training_params.get('scheduler_kwargs', {})
            scheduler = scheduler_class(optimizer, **scheduler_kwargs)
        
        return optimizer, scheduler
    
    def _setup_trainer(self,
                      model: mz.models.Model,
                      optimizer: torch.optim.Optimizer,
                      trainloader: mz.dataloader.DataLoader,
                      testloader: mz.dataloader.DataLoader,
                      epochs: int,
                      save_dir: Path,
                      config: Dict[str, Any],
                      scheduler: Optional[Any] = None) -> mz.trainers.Trainer:
        """Setup trainer"""
        training_params = config['training_params']
        
        trainer_kwargs = {
            'model': model,
            'optimizer': optimizer,
            'trainloader': trainloader,
            'validloader': testloader,
            'validate_interval': None,
            'epochs': epochs,
            'device': self.device,
            'save_dir': str(save_dir)
        }
        
        if scheduler:
            trainer_kwargs['scheduler'] = scheduler
        
        if 'clip_norm' in training_params:
            trainer_kwargs['clip_norm'] = training_params['clip_norm']
        
        return mz.trainers.Trainer(**trainer_kwargs)

# ===== CLI Functions =====
def parse_arguments():
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(
        description="MatchZoo Training Controller với CLI interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train một model
  python train_controller.py --model ArcII --data-pack /path/to/datapack --epochs 10
  
  # Train nhiều models
  python train_controller.py --model ArcII KNRM ESIM --data-pack /path/to/datapack
  
  # Custom parameters
  python train_controller.py --model ArcII --data-pack /path/to/datapack --batch-size 32 --lr 0.001
  
  # GPU training  
  python train_controller.py --model ArcII --data-pack /path/to/datapack --device cuda
        """
    )
    
    parser.add_argument(
        '--model', '--models',
        nargs='+',
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help='Model(s) cần huấn luyện'
    )
    
    parser.add_argument(
        '--data-pack',
        required=True,
        help='Đường dẫn đến thư mục chứa train_pack.dam và test_pack.dam'
    )
    
    parser.add_argument(
        '--output-dir',
        default='Trained_model',
        help='Thư mục output để lưu models (default: Trained_model)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Số epochs để train (default: 5)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (override default của từng model)'
    )
    
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        help='Learning rate (override default của từng optimizer)'
    )
    
    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'dml', 'cpu'],
        default='auto',
        help='Device để training (default: auto)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='In chi tiết quá trình training'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Tắt verbose output'
    )
    
    return parser.parse_args()

def main():
    """Main CLI entry point"""
    args = parse_arguments()
    
    # Handle quiet mode
    verbose = args.verbose and not args.quiet
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(
            data_pack_dir=args.data_pack,
            output_dir=args.output_dir,
            device=args.device,
            verbose=verbose
        )
        
        trained_models = []
        
        # Train each model
        for model_name in args.model:
            if verbose:
                print(f"\n{'='*60}")
                print(f"TRAINING {model_name}")
                print(f"{'='*60}")
            
            try:
                model_path = trainer.train_model(
                    model_name=model_name,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr
                )
                trained_models.append((model_name, model_path))
                
            except Exception as e:
                print(f"Lỗi khi train {model_name}: {e}")
                continue
        
        # Summary
        if verbose and trained_models:
            print(f"\n{'='*60}")
            print("TRAINING HOÀN THÀNH")
            print(f"{'='*60}")
            print(f"Đã train thành công {len(trained_models)}/{len(args.model)} models:")
            for model_name, model_path in trained_models:
                print(f"   {model_name}: {model_path}")
            print(f"\nSử dụng evaluate_models.py để đánh giá performance!")
        
    except Exception as e:
        print(f"Lỗi: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
