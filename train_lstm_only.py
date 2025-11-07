import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Concatenate, 
    Flatten, Attention, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print("--- ۱. بارگذاری داده‌های پردازش شده ---")

try:
    data = np.load('processed_hybrid_data.npz')
except FileNotFoundError:
    print("فایل 'processed_hybrid_data.npz' پیدا نشد.")
    print("لطفاً ابتدا اسکریپت build_dataset.py را اجرا کنید.")
    exit()

# --- تغییر: ما دیگر به X_user نیازی نداریم ---
# X_user = data['X_user'] 
X_movie = data['X_movie']
X_time = data['X_time']
y = data['y']

n_users = int(data['n_users'])
n_movies = int(data['n_movies'])
n_time_features = int(data['n_time_features'])

MAX_SEQUENCE_LENGTH = X_movie.shape[1] 

print(f"تعداد فیلم‌ها: {n_movies}")
print(f"تعداد دسته‌های زمانی: {n_time_features}")
print(f"حداکثر طول دنباله: {MAX_SEQUENCE_LENGTH}")


print("\n--- ۲. تقسیم داده‌ها به آموزشی و آزمایشی ---")

# --- تغییر: ورودی X_user از لیست‌ها حذف شد ---
indices = np.arange(y.shape[0])
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

# X_train و X_test حالا فقط شامل ۲ ورودی هستند
X_train = [X_movie[train_indices], X_time[train_indices]]
X_test = [X_movie[test_indices], X_time[test_indices]]

y_train = y[train_indices]
y_test = y[test_indices]

print(f"تعداد نمونه‌های آموزشی: {len(y_train)}")
print(f"تعداد نمونه‌های آزمایشی: {len(y_test)}")


print("\n--- ۳. ساخت معماری مدل (فقط LSTM + Attention) ---")


# --- تعریف هایپرپارامترها ---
# (بدون تغییر)
MOVIE_EMBEDDING_SIZE = 64
TIME_EMBEDDING_SIZE = 16
LSTM_UNITS = 64

# --- تعریف ورودی‌ها ---
# --- تغییر: ورودی کاربر حذف شد ---
movie_seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='MovieSequenceInput')
time_seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='TimeSequenceInput')


# --- شاخه ۱: پروفایل کاربر (NCF) ---
# --- تغییر: کل این شاخه حذف شد ---


# --- شاخه ۲: رفتار ترتیبی (LSTM + Time + Attention) ---
# (این بخش بدون تغییر باقی می‌ماند)
movie_embedding_layer = Embedding(input_dim=n_movies, output_dim=MOVIE_EMBEDDING_SIZE, name='MovieEmbedding')
movie_seq_embedding = movie_embedding_layer(movie_seq_input)

time_embedding_layer = Embedding(input_dim=n_time_features, output_dim=TIME_EMBEDDING_SIZE, name='TimeEmbedding')
time_seq_embedding = time_embedding_layer(time_seq_input)

combined_seq_embedding = Concatenate(axis=2, name='CombineMovieTime')([movie_seq_embedding, time_seq_embedding])
lstm_out = LSTM(LSTM_UNITS, return_sequences=True, name='LSTM')(combined_seq_embedding)
attention_out = Attention(name='Attention')([lstm_out, lstm_out])
attention_vec = GlobalAveragePooling1D(name='AttentionPooling')(attention_out)


# --- ادغام نهایی ---
# --- تغییر: دیگر ادغامی وجود ندارد. خروجی Attention مستقیماً به MLP می‌رود ---
final_vec = attention_vec 

# --- شبکه MLP نهایی ---
dense_1 = Dense(128, activation='relu', name='Dense_1')(final_vec) # ورودی از final_vec است
dense_2 = Dense(64, activation='relu', name='Dense_2')(dense_1)
output = Dense(n_movies, activation='softmax', name='Output')(dense_2)


# --- ساخت مدل نهایی ---
# --- تغییر: مدل حالا فقط ۲ ورودی دارد ---
model = Model(
    inputs=[movie_seq_input, time_seq_input], 
    outputs=output,
    name='LSTM_Only_Model' # <-- نام مدل تغییر کرد
)

model.summary()


print("\n--- ۴. کامپایل و آموزش کامل مدل ---")

metrics = [
    'sparse_categorical_accuracy', 
    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc')
]

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy', 
    metrics=metrics
)

# (EarlyStopping بدون تغییر است)
early_stopper = EarlyStopping(
    monitor='val_top_10_acc', 
    patience=3, 
    verbose=1,
    mode='max',
    restore_best_weights=True 
)

print("شروع آموزش مدل LSTM-Only (تا 100 دوره)...")

history = model.fit(
    X_train, y_train,
    batch_size=256,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stopper]
)

print("\n--- ۵. ارزیابی نهایی مدل ---")
results = model.evaluate(X_test, y_test)

print(f"✅ مدل LSTM-Only با موفقیت آموزش دید.")
print(f"Loss (خطا) روی داده‌های تست: {results[0]:.4f}")
print(f"Accuracy (دقت) روی داده‌های تست: {results[1] * 100:.2f}%")
print(f"Top 10 Accuracy روی داده‌های تست: {results[2] * 100:.2f}%")

# --- تغییر: نام فایل ذخیره شده ---
model.save('lstm_only_model.keras')
print("مدل LSTM-Only با موفقیت در فایل 'lstm_only_model.keras' ذخیره شد.")