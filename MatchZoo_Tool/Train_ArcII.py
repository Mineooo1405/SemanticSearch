import matchzoo as mz 
import numpy as np
import torch
import os
import gc

print("--- Bắt đầu huấn luyện model Arc-II ---")

# --- A. Cấu hình Device ---
device = torch.device("cpu")
print(f"Đã chọn device: {device}.")

# --- 1. Tải các DataPack ---
data_pack_dir = input("Nhập đường dẫn đến thư mục chứa các DataPack: ")
train_pack_raw = mz.load_data_pack(os.path.join(data_pack_dir, 'train_pack.dam'))
test_pack_raw = mz.load_data_pack(os.path.join(data_pack_dir, 'test_pack.dam'))
print("Tải DataPacks thành công.")

# --- 2. Thiết lập Task và Metrics ---
ranking_task = mz.tasks.Ranking()
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]

# --- 3. Tiền xử lý dữ liệu ---
preprocessor = mz.models.ArcII.get_default_preprocessor(
    filter_mode='df',
    filter_low_freq=2,
)
train_pack_processed = preprocessor.fit_transform(train_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)
print("Tiền xử lý dữ liệu hoàn tất.")

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
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=2,
    num_neg=1,
    batch_size=20, 
    resample=True,
    sort=False,
    shuffle=True
)
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed,
    mode='point', 
    batch_size=20, 
    shuffle=False
)

padding_callback = mz.models.ArcII.get_default_padding_callback(
    fixed_length_left=10,
    fixed_length_right=256,  
    pad_word_value=0,
    pad_word_mode='pre'
)

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
model = mz.models.ArcII()
model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['left_length'] = 10
model.params['right_length'] = 256  
model.params['kernel_1d_count'] = 32
model.params['kernel_1d_size'] = 3
model.params['kernel_2d_count'] = [64, 64]
model.params['kernel_2d_size'] = [(3, 3), (3, 3)]
model.params['pool_2d_size'] = [(3, 3), (3, 3)]
model.params['dropout_rate'] = 0.3
model.build()
model.to(device)
print(model)
print("Tổng số tham số cần huấn luyện:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# --- 7. Huấn luyện Model ---
optimizer = torch.optim.Adam(model.parameters())
trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=testloader,
    validate_interval=None,
    epochs=5,
    device=device,
    save_dir='Trained_model/arcii_model'
)
trainer.run()

# --- 8. Lưu model và preprocessor ---
print("Bắt đầu lưu model và preprocessor...")
trainer.save_model()
preprocessor.save('Trained_model/arcii_model/preprocessor')
print("Đã lưu model và preprocessor vào thư mục 'arcii_model'.")
