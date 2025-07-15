from typing import cast
import pandas as pd
import matchzoo as mz

# 1) Đọc file TSV
path = r'F:\SematicSearch\training_datasets\semantic_grouping_with_OIE\output_chunks_final_20250712_062321_rrf_filtered_3col.tsv'
# Đọc TSV. Nếu file không có header, đặt header mặc định
df_raw = cast(pd.DataFrame, pd.read_csv(path, sep='\t', header=None))  # type: ignore[arg-type]

# Nếu cột chưa được đặt tên ('query' không xuất hiện) → gán thủ công
if 'query' not in df_raw.columns:
    df_raw.columns = ['query', 'passage', 'label']

# 2) Chuẩn hoá tên cột
# ép kiểu DataFrame để linter hiểu rõ
df = cast(pd.DataFrame, df_raw.rename(columns={
    'query':   'text_left',
    'passage': 'text_right',
    'label':   'label'
}))

# Loại bỏ các dòng header giả (label = 'label') và ép kiểu số
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df.dropna(subset=['text_left', 'text_right', 'label'], inplace=True)
# reset index và loại mọi NaN còn lại
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# 3) Đóng gói thành DataPack (MatchZoo tự sinh id_left / id_right)
basic_df = df[['text_left', 'text_right', 'label']]
pack = mz.pack(basic_df)

# 6) (tuỳ chọn) Lưu lại để lần sau load nhanh
pack.save('robust04_pack.dam')
print('DataPack kích thước:', len(pack))
print(pack.frame().head())

# Chia train/test
import numpy as np
np.random.seed(42)
pack = cast(mz.DataPack, pack.shuffle())  # shuffle và nhận bản mới
split = int(len(pack)*0.8)
train_pack = pack[:split]
test_pack  = pack[split:]

# ------------------- Huấn luyện với auto.prepare ---------------------------

# Thiết lập task Ranking kèm nhiều metrics
task = mz.tasks.Ranking()
task.metrics = [
    mz.metrics.MeanAveragePrecision(),
    mz.metrics.Precision(k=1),
    mz.metrics.Precision(k=5),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=10),
]
# Bạn có thể đổi sang bất kỳ model nào trong danh sách mz.models.*
MODEL_CLASS = mz.models.DRMM  # ví dụ: mz.models.KNRM, mz.models.DRMM, v.v.

# auto.prepare trả về: model, preprocessor, data_generator_builder, embedding_matrix
model, preproc, dg_builder, _ = mz.auto.prepare(
    task=task,
    model_class=MODEL_CLASS,
    data_pack=train_pack,
)

# Tiền xử lý & tạo DataGenerator
train_processed = preproc.transform(train_pack, verbose=0)
test_processed  = preproc.transform(test_pack,  verbose=0)

train_gen = dg_builder.build(train_processed, batch_size=100)
test_gen  = dg_builder.build(test_processed, batch_size=100)
# Huấn luyện
model.fit_generator(train_gen, epochs=50)  # type: ignore[misc]

# Đánh giá
print("Evaluation:", model.evaluate_generator(test_gen))  # type: ignore[misc]
# Lưu lại mô hình và preprocessor để dùng lại sau này
model.save('drmm_model')
preproc.save('drmm_preprocessor')