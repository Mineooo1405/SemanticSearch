import matchzoo as mz 
import numpy as np
import torch
import os
import gc
import pandas as pd

print("--- Bắt đầu huấn luyện model Conv-KNRM ---")

# --- A. Cấu hình Device ---
# Cố gắng sử dụng DirectML, sau đó CUDA, cuối cùng CPU
try:
    import torch_directml
    device = torch_directml.device()
    print("Đã tìm thấy và chọn device: DirectML (AMD GPU).")
except (ImportError, AttributeError):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DirectML không khả dụng. Đã chọn device: {device}")

# --- 1. Tải các DataPack đã được tạo ---
data_pack_dir = input("Nhập đường dẫn đến thư mục chứa các DataPack (thư mục gốc có cv_folds/ hoặc chứa train_pack.dam/test_pack.dam): ").strip()

train_path = os.path.join(data_pack_dir, 'train_pack.dam')
test_path = os.path.join(data_pack_dir, 'test_pack.dam')

# Hỗ trợ Cross-Validation: nếu có cv_folds thì hỏi fold số mấy hoặc tự chọn nếu chỉ có 1 fold
cv_dir = os.path.join(data_pack_dir, 'cv_folds')
if os.path.isdir(cv_dir):
    fold_files = sorted([f for f in os.listdir(cv_dir) if f.startswith('fold_') and (f.endswith('_train.dam') or f.endswith('_test.dam'))])
    # Lấy danh sách fold số
    folds = sorted({int(name.split('_')[1]) for name in fold_files if name.split('_')[1].isdigit()})
    if not folds:
        print(f"Không tìm thấy files fold_* trong {cv_dir}")
        raise SystemExit(1)
    if len(folds) == 1:
        fold = folds[0]
        print(f"Phát hiện 1 fold: dùng fold {fold}")
    else:
        print(f"Phát hiện các folds: {folds}")
        try:
            fold = int(input(f"Nhập số fold để train [mặc định {folds[0]}]: ") or folds[0])
        except Exception:
            fold = folds[0]
    train_path = os.path.join(cv_dir, f"fold_{fold}_train.dam")
    test_path = os.path.join(cv_dir, f"fold_{fold}_test.dam")

if not os.path.isfile(train_path) or not os.path.isfile(test_path):
    print(f"Không tìm thấy file DataPack:\n  train: {train_path}\n  test : {test_path}")
    raise SystemExit(1)

print(f"Sử dụng train: {train_path}")
print(f"Sử dụng test : {test_path}")

train_pack_raw = mz.load_data_pack(train_path)
test_pack_raw = mz.load_data_pack(test_path)

print("Tải DataPacks thành công.")

# --- 2. Thiết lập Task và Metrics ---
ranking_task = mz.tasks.Ranking()  # ConvKNRM dùng loss mặc định
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

# --- Chẩn đoán: kích thước pack và phân phối nhãn trước/sau preprocessor ---
def _print_pack_stats(title: str, pack):
    try:
        size = len(pack)
    except Exception:
        size = -1
    has_rel = hasattr(pack, 'relation') and getattr(pack, 'relation') is not None
    print(f"[{title}] size={size}, relation={'yes' if has_rel else 'no'}")
    if has_rel:
        try:
            rel = pack.relation
            print(f"[{title}] relation rows={len(rel)} label_dist={rel['label'].value_counts(dropna=False).to_dict()}")
        except Exception as e:
            print(f"[{title}] không thể in label_dist: {e}")

_print_pack_stats('RAW TRAIN', train_pack_raw)
_print_pack_stats('PROC TRAIN', train_pack_processed)

# --- 3.1. Kiểm tra khả năng tạo cặp pos/neg cho training ---
def _check_pairable(pack) -> bool:
    try:
        rel = getattr(pack, 'relation', None)
        if rel is None or rel.empty:
            print("[Cảnh báo] DataPack không có relation hoặc rỗng.")
            return False
        # label > 0 coi là positive, else negative
        stats = rel.groupby('id_left')['label'].agg(pos=lambda s: (s > 0).sum(), neg=lambda s: (s <= 0).sum())
        total_left = stats.shape[0]
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

# --- 4. Chuẩn bị Embedding Matrix ---
glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=100)
term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = glove_embedding.build_matrix(term_index)
l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
l2_norm[l2_norm == 0] = 1e-8
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

print("Tạo Embedding Matrix thành công.")
del glove_embedding
gc.collect()
print("Đã giải phóng bộ nhớ từ GloVe embedding.")


# --- 5. Tạo DataLoader ---
# Mặc định dùng pairwise; fallback sang point nếu không tạo được cặp
use_pairwise = _check_pairable(train_pack_processed)
try:
    if use_pairwise:
        trainset = mz.dataloader.Dataset(
            data_pack=train_pack_processed,
            mode='pair',
            num_dup=5,
            num_neg=1,
            batch_size=20,
            resample=True,
            sort=False,
            shuffle=True
        )
        print("Trainset: pairwise (num_dup=5, num_neg=1)")
    else:
        raise ValueError("Pairwise not feasible; fallback to pointwise")
except Exception as e:
    print(f"[Cảnh báo] Không thể tạo trainset pairwise ({e}). Chuyển sang pointwise.")
    trainset = mz.dataloader.Dataset(
        data_pack=train_pack_processed,
        mode='point',
        batch_size=20,
        shuffle=True
    )

testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed,
    mode='point',
    batch_size=20,
    shuffle=False
)

padding_callback = mz.models.ConvKNRM.get_default_padding_callback()

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
model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['filters'] = 128
model.params['conv_activation_func'] = 'tanh'
model.params['max_ngram'] = 3
model.params['use_crossmatch'] = True
model.params['kernel_num'] = 11
model.params['sigma'] = 0.1
model.params['exact_sigma'] = 0.001

model.build()
try:
    model.to(device)
except Exception as e:
    print(f"[Cảnh báo] Không thể chuyển model sang device {device}: {e}\n  Fallback sang CPU.")
    device = torch.device('cpu')
    # Rebuild DataLoaders to use CPU device
    padding_callback = mz.models.ConvKNRM.get_default_padding_callback()
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
    model.to(device)

print(model)
print("Tổng số tham số cần huấn luyện:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# --- 7. Huấn luyện Model ---
optimizer = torch.optim.Adadelta(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)

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

trainer.run()

# --- 8. Lưu model và preprocessor ---
print("Bắt đầu lưu model và preprocessor...")
trainer.save_model()
preprocessor.save('Trained_model/conv_knrm_model/preprocessor')
print("Đã lưu model và preprocessor vào thư mục 'conv_knrm_model'.")
