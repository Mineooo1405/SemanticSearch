import os
# Help CUDA memory allocator avoid fragmentation before torch loads.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import matchzoo as mz 
import numpy as np
import torch
import gc
import pandas as pd
import nltk
nltk.download('punkt')
print("--- Bắt đầu huấn luyện model Conv-KNRM ---")

# --- A. Cấu hình Device ---
# Cố gắng sử dụng DirectML
force_device = os.environ.get("FORCE_DEVICE")  # e.g., "cpu", "cuda", or "cuda:0"
if force_device:
    device = torch.device(force_device)
    print(f"FORCE_DEVICE được đặt, sử dụng: {device}")
else:
    try:
        import torch_directml
        device = torch_directml.device()
        print("Đã tìm thấy và chọn device: DirectML (AMD GPU).")
    except (ImportError, AttributeError):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DirectML không khả dụng. Đã chọn device: {device}")
# --- 1. Tải các DataPack đã được tạo ---
# Trỏ đến thư mục chứa các file train_pack.dam, dev_pack.dam, test_pack.dam

train_pack_raw = mz.load_data_pack("/workspace/cv_folds/fold_1_train.dam")
test_pack_raw = mz.load_data_pack("/workspace/cv_folds/fold_1_test.dam")

print("Tải DataPacks thành công.")

# --- 2. Thiết lập Task và Metrics ---
ranking_task = mz.tasks.Ranking() # ConvKNRM dùng loss mặc định
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]

# --- 3. Tiền xử lý dữ liệu ---
preprocessor = mz.models.ConvKNRM.get_default_preprocessor()

train_pack_processed = preprocessor.fit_transform(train_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)

print("Tiền xử lý dữ liệu hoàn tất.")

# --- 3.1. Kiểm tra khả năng tạo cặp pos/neg cho training (để tránh lỗi concat rỗng) ---
def _check_pairable(pack) -> bool:
    try:
        rel = getattr(pack, 'relation', None)
        if rel is None or rel.empty:
            print("[Cảnh báo] DataPack không có relation hoặc rỗng.")
            return False
        # label > 0 coi là positive, else negative (tránh lỗi pandas aggregate named-agg trên SeriesGroupBy)
        df_tmp = rel.copy()
        df_tmp['pos'] = (df_tmp['label'] > 0).astype(int)
        df_tmp['neg'] = (df_tmp['label'] <= 0).astype(int)
        stats = df_tmp.groupby('id_left')[['pos', 'neg']].sum()
        total_left = int(stats.shape[0])
        valid_left = (stats['pos'] > 0) & (stats['neg'] > 0)
        n_valid = int(valid_left.sum())
        print(f"Phân phối nhãn (train): total_left={total_left}, left_có_pos&neg={n_valid}")
        if n_valid == 0:
            print("[Cảnh báo] Không có left id nào có cả positive và negative. Pairwise sẽ thất bại.")
            return False
        return True
    except Exception as e:
        print(f"[Cảnh báo] Lỗi kiểm tra pairable: {e}")
        return False

# In thống kê nhanh để chẩn đoán trên dataset lớn/nhỏ
def _print_pack_stats(prefix: str, pack):
    try:
        rel = getattr(pack, 'relation', None)
        if rel is None or rel.empty:
            print(f"[{prefix}] relation: EMPTY")
            return
        total = int(len(rel))
        pos = int((rel['label'] > 0).sum())
        neg = total - pos
        nl = int(rel['id_left'].nunique())
        nr = int(rel['id_right'].nunique())
        print(f"[{prefix}] rows={total} pos={pos} neg={neg} id_left={nl} id_right={nr}")
    except Exception as e:
        print(f"[{prefix}] Stats error: {e}")

# --- 4. Chuẩn bị Embedding Matrix ---
# Tải GloVe embedding
glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=100)
# Lấy từ điển từ preprocessor
term_index = preprocessor.context['vocab_unit'].state['term_index']
# Xây dựng ma trận embedding cho các từ trong bộ dữ liệu
embedding_matrix = glove_embedding.build_matrix(term_index)
# Chuẩn hóa L2
l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
# Thêm một số epsilon nhỏ để tránh chia cho 0
l2_norm[l2_norm == 0] = 1e-8 
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

print("Tạo Embedding Matrix thành công.")

# Giải phóng bộ nhớ RAM
del glove_embedding
gc.collect()
print("Đã giải phóng bộ nhớ từ GloVe embedding.")

_print_pack_stats("RAW TRAIN", train_pack_raw)
_print_pack_stats("PROC TRAIN", train_pack_processed)
_print_pack_stats("RAW TEST", test_pack_raw)
_print_pack_stats("PROC TEST", test_pack_processed)


# --- 5. Tạo DataLoader ---
# Mặc định dùng pairwise; fallback sang point nếu không tạo được cặp
use_pairwise = _check_pairable(train_pack_processed)
try:
    if use_pairwise:
        trainset = mz.dataloader.Dataset(
            data_pack=train_pack_processed,
            mode='pair',
            num_dup=2,
            num_neg=1,
            batch_size=8,
            resample=True,
            sort=False,
            shuffle=True
        )
        print("Trainset: pairwise (num_dup=2, num_neg=1)")
    else:
        raise ValueError("Pairwise not feasible; fallback to pointwise")
except Exception as e:
    print(f"[Cảnh báo] Không thể tạo trainset pairwise ({e}). Chuyển sang pointwise.")
    trainset = mz.dataloader.Dataset(
        data_pack=train_pack_processed,
        mode='point',
        batch_size=8,
        shuffle=True
    )
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed,
    mode='point', 
    batch_size=8, 
    shuffle=False
)

# Lấy callback để padding
padding_callback = mz.models.ConvKNRM.get_default_padding_callback()

# Tạo DataLoader
trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback,
    device=device
)
testloader = mz.dataloader.DataLoader(
    dataset=testset,
    stage='dev',
    callback=padding_callback,
    device=device
)

print("Tạo DataLoader thành công.")

# --- 6. Xây dựng và cấu hình Model ---
model = mz.models.ConvKNRM()

# Gán các tham số cho model
model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['filters'] = 64 
model.params['conv_activation_func'] = 'tanh' 
model.params['max_ngram'] = 3
model.params['use_crossmatch'] = True 
model.params['kernel_num'] = 7
model.params['sigma'] = 0.1
model.params['exact_sigma'] = 0.001

# Xây dựng kiến trúc model
model.build()

# Sanity check: expected feature dimension vs model.out layer
try:
    _use_cross = model.params['use_crossmatch']
    _knum = model.params['kernel_num']
    _ng = model.params['max_ngram']
except Exception:
    _use_cross = True
    _knum = 7
    _ng = 3
_expected_phi = _knum * ((_ng * _ng) if _use_cross else _ng)
try:
    _linear_in = model.out.in_features  # type: ignore[attr-defined]
    print(f"[Debug] expected_phi={_expected_phi}, linear_in_features={_linear_in}, use_crossmatch={_use_cross}")
except Exception:
    print(f"[Debug] expected_phi={_expected_phi}, use_crossmatch={_use_cross}")

# Chuyển model sang device đã chọn
model.to(device)

print(model)
print("Tổng số tham số cần huấn luyện:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# --- 7. Huấn luyện Model ---
# Chọn optimizer
optimizer = torch.optim.Adadelta(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)

# Thiết lập Trainer
trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=testloader,
    validate_interval=None,
    epochs=5,
    device=device,
    save_dir='Trained_model/conv_knrm_model',
    scheduler=scheduler,
    clip_norm=10
)

# Bắt đầu huấn luyện
try:
    trainer.run()
except torch.cuda.OutOfMemoryError as oom:
    print(f"[Cảnh báo] Hết bộ nhớ GPU: {oom}. Thử giải phóng cache và giảm tải.")
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    print("Gợi ý: giảm batch_size thêm, hoặc chạy trên CPU bằng cách tắt CUDA.")
    raise

# --- 8. Lưu model và preprocessor ---
print("Bắt đầu lưu model và preprocessor...")
trainer.save_model() # Sử dụng trainer để lưu model
preprocessor.save('Trained_model/conv_knrm_model/preprocessor') # Lưu preprocessor vào cùng thư mục
print("Đã lưu model và preprocessor vào thư mục 'Trained_model/conv_knrm_model'.")
