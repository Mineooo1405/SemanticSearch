import matchzoo as mz 
import numpy as np
import torch
import os
import gc

print("--- Bắt đầu huấn luyện model MatchPyramid ---")

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
preprocessor = mz.models.MatchPyramid.get_default_preprocessor()
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

# --- 5. Tạo DataLoader với batch size lớn hơn ---
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=5,
    num_neg=1,
    batch_size=64,  # Tăng từ 16 lên 64
    resample=True,
    sort=False,
    shuffle=True
)
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed,
    mode='point', 
    batch_size=64,  # Tăng từ 16 lên 64
    shuffle=False
)

padding_callback = mz.models.MatchPyramid.get_default_padding_callback()

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

# --- 6. Xây dựng và cấu hình Model với hyperparameters tùy chỉnh ---
model = mz.models.MatchPyramid()
model.params['task'] = ranking_task

# Cấu hình hyperparameters như đã thảo luận
model.params['kernel_count'] = [16, 32]  # Hai lớp conv với 16 và 32 kernels
model.params['kernel_size'] = [[3, 3], [3, 3]]  # Kernel size 3x3 cho cả hai lớp
model.params['activation'] = 'relu'  # Giữ ReLU activation
model.params['dpool_size'] = [3, 10]  # Dynamic pooling size
model.params['dropout_rate'] = 0.2  # Thêm dropout để tránh overfitting

model.params['embedding'] = embedding_matrix
model.build()
model.to(device)
print(model)
print("Tổng số tham số cần huấn luyện:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# --- 7. Huấn luyện Model với learning rate tùy chỉnh ---
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Đặt rõ learning rate

# Thêm learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',  # Monitor validation metric (cao hơn là tốt hơn)
    factor=0.5,  # Giảm LR xuống một nửa
    patience=3,  # Chờ 3 epochs không cải thiện
    verbose=True
)

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=testloader,
    validate_interval=1,  # Validate mỗi epoch
    epochs=20,  # Tăng từ 5 lên 20 epochs
    device=device,
    save_dir='Trained_model/match_pyramid_model'
)

# Thêm callback để monitor training
class LRSchedulerCallback:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        
    def on_epoch_end(self, trainer):
        # Lấy validation score từ trainer
        if trainer.validate_interval and len(trainer.state.val_scores) > 0:
            current_score = trainer.state.val_scores[-1]
            self.scheduler.step(current_score)

trainer.run()

# --- 8. Lưu model và preprocessor ---
print("Bắt đầu lưu model và preprocessor...")
trainer.save_model()
preprocessor.save('Trained_model/match_pyramid_model/preprocessor')
print("Đã lưu model và preprocessor vào thư mục 'match_pyramid_model'.")

# --- 9. Thêm đánh giá cuối cùng ---
print("\n--- Đánh giá cuối cùng trên test set ---")
final_results = trainer.evaluate(testloader)
print("Kết quả cuối cùng:", final_results)
