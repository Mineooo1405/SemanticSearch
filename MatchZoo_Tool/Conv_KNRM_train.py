import matchzoo as mz 
import numpy as np
import torch
import os
import gc

print("--- Bắt đầu huấn luyện model Conv-KNRM ---")

# --- A. Cấu hình Device ---
# Cố gắng sử dụng DirectML
try:
    import torch_directml
    device = torch_directml.device()
    print("Đã tìm thấy và chọn device: DirectML (AMD GPU).")
except (ImportError, AttributeError):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DirectML không khả dụng. Đã chọn device: {device}")

device = torch.device("cpu")
# --- 1. Tải các DataPack đã được tạo ---
# Trỏ đến thư mục chứa các file train_pack.dam, dev_pack.dam, test_pack.dam
data_pack_dir = input("Nhập đường dẫn đến thư mục chứa các DataPack: ")

train_pack_raw = mz.load_data_pack(os.path.join(data_pack_dir, 'train_pack.dam'))
dev_pack_raw = mz.load_data_pack(os.path.join(data_pack_dir, 'dev_pack.dam'))
test_pack_raw = mz.load_data_pack(os.path.join(data_pack_dir, 'test_pack.dam'))

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
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)

print("Tiền xử lý dữ liệu hoàn tất.")

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


# --- 5. Tạo DataLoader ---
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=5,
    num_neg=1, # ConvKNRM thường dùng 1 neg
    batch_size=16, 
    resample=True,
    sort=False,
    shuffle=True
)
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed,
    mode='point', 
    batch_size=16, 
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

# Xây dựng kiến trúc model
model.build()

# Chuyển model sang device đã chọn
model.to(device)

print(model)
print("Tổng số tham số cần huấn luyện:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# --- 7. Huấn luyện Model ---
# Chọn optimizer
optimizer = torch.optim.Adam(model.parameters())

# Thiết lập Trainer
trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=testloader,
    validate_interval=None,
    epochs=5,
    device=device,
    save_dir='conv_knrm_model' # Chỉ định thư mục để lưu model
)

# Bắt đầu huấn luyện
trainer.run()

# --- 8. Lưu model và preprocessor ---
print("Bắt đầu lưu model và preprocessor...")
trainer.save_model() # Sử dụng trainer để lưu model
preprocessor.save('conv_knrm_model/preprocessor') # Lưu preprocessor vào cùng thư mục
print("Đã lưu model và preprocessor vào thư mục 'conv_knrm_model'.")
