import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Concatenate, 
    Flatten, Attention, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
# ۱. وارد کردن EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping

print("--- ۱. بارگذاری داده‌های پردازش شده ---")

# بارگذاری فایل NPZ
try:
    data = np.load('processed_hybrid_data.npz')
except FileNotFoundError:
    print("فایل 'processed_hybrid_data.npz' پیدا نشد.")
    print("لطفاً ابتدا اسکریپت build_dataset.py را اجرا کنید.")
    exit()

X_user = data['X_user']
X_movie = data['X_movie']
X_time = data['X_time']
y = data['y']

n_users = int(data['n_users'])
n_movies = int(data['n_movies'])
n_time_features = int(data['n_time_features'])

# دریافت طول دنباله از داده‌ها
MAX_SEQUENCE_LENGTH = X_movie.shape[1] 

print(f"تعداد کاربران: {n_users}")
print(f"تعداد فیلم‌ها: {n_movies}")
print(f"تعداد دسته‌های زمانی: {n_time_features}")
print(f"حداکثر طول دنباله: {MAX_SEQUENCE_LENGTH}")


print("\n--- ۲. تقسیم داده‌ها به آموزشی و آزمایشی ---")

# تقسیم داده‌ها
indices = np.arange(X_user.shape[0])
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

X_train = [X_user[train_indices], X_movie[train_indices], X_time[train_indices]]
X_test = [X_user[test_indices], X_movie[test_indices], X_time[test_indices]]

y_train = y[train_indices]
y_test = y[test_indices]

print(f"تعداد نمونه‌های آموزشی: {len(y_train)}")
print(f"تعداد نمونه‌های آزمایشی: {len(y_test)}")


print("\n--- ۳. ساخت معماری مدل هیبریدی (NCF + LSTM + Attention) ---")

# --- تعریف هایپرپارامترها ---
USER_EMBEDDING_SIZE = 64
MOVIE_EMBEDDING_SIZE = 64
TIME_EMBEDDING_SIZE = 16
LSTM_UNITS = 64

# --- تعریف ورودی‌ها ---
user_input = Input(shape=(1,), name='UserInput')
movie_seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='MovieSequenceInput')
time_seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='TimeSequenceInput')


# --- شاخه ۱: پروفایل کاربر (NCF) ---
user_embedding_layer = Embedding(input_dim=n_users, output_dim=USER_EMBEDDING_SIZE, name='UserEmbedding')
user_vec = Flatten(name='FlattenUser')(user_embedding_layer(user_input))


# --- شاخه ۲: رفتار ترتیبی (LSTM + Time + Attention) ---
movie_embedding_layer = Embedding(input_dim=n_movies, output_dim=MOVIE_EMBEDDING_SIZE, name='MovieEmbedding')
movie_seq_embedding = movie_embedding_layer(movie_seq_input)

time_embedding_layer = Embedding(input_dim=n_time_features, output_dim=TIME_EMBEDDING_SIZE, name='TimeEmbedding')
time_seq_embedding = time_embedding_layer(time_seq_input)

combined_seq_embedding = Concatenate(axis=2, name='CombineMovieTime')([movie_seq_embedding, time_seq_embedding])
lstm_out = LSTM(LSTM_UNITS, return_sequences=True, name='LSTM')(combined_seq_embedding)
attention_out = Attention(name='Attention')([lstm_out, lstm_out])
attention_vec = GlobalAveragePooling1D(name='AttentionPooling')(attention_out)


# --- ادغام نهایی دو شاخه ---
final_combined_vec = Concatenate(name='CombineNCF_LSTM')([user_vec, attention_vec])

# --- شبکه MLP نهایی ---
dense_1 = Dense(128, activation='relu', name='Dense_1')(final_combined_vec)
dense_2 = Dense(64, activation='relu', name='Dense_2')(dense_1)
output = Dense(n_movies, activation='softmax', name='Output')(dense_2)


# --- ساخت مدل نهایی ---
model = Model(
    inputs=[user_input, movie_seq_input, time_seq_input], 
    outputs=output,
    name='Hybrid_NCF_LSTM_Attention_Model'
)

# نمایش خلاصه معماری
model.summary()


print("\n--- ۴. کامپایل و آموزش کامل مدل ---")

# تعریف معیارها
metrics = [
    'sparse_categorical_accuracy', 
    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc')
]

# کامپایل مدل
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy', 
    metrics=metrics
)

# --- ۲. تعریف EarlyStopping ---
# ما معیار 'val_top_10_acc' را زیر نظر می‌گیریم (دقت Top 10 روی داده‌های تست)
# 'patience=3' یعنی 3 دوره صبر می‌کند تا ببیند آیا معیار بهتر می‌شود یا نه
# 'mode='max'' یعنی ما به دنبال بیشینه کردن این معیار هستیم
early_stopper = EarlyStopping(
    monitor='val_top_10_acc', 
    patience=3, 
    verbose=1, # این خط، زمان توقف را به شما اطلاع می‌دهد
    mode='max',
    restore_best_weights=True # این خط، بهترین وزن‌ها را به مدل برمی‌گرداند
)

print("شروع آموزش کامل مدل (تا 100 دوره یا توقف زودهنگام)...")

# --- ۳. آموزش نهایی با Epochs=100 و EarlyStopping ---
history = model.fit(
    X_train, y_train,
    batch_size=256,
    epochs=100,       # حداکثر تعداد دورها
    validation_data=(X_test, y_test),
    callbacks=[early_stopper] # ناظر را به آموزش اضافه می‌کنیم
)

print("\n--- ۵. ارزیابی نهایی مدل ---")
# ارزیابی با بهترین وزن‌هایی که EarlyStopping پیدا کرده
results = model.evaluate(X_test, y_test)

print(f"✅ مرحله نهایی با موفقیت انجام شد.")
print(f"Loss (خطا) روی داده‌های تست: {results[0]:.4f}")
print(f"Accuracy (دقت) روی داده‌های تست: {results[1] * 100:.2f}%")
print(f"Top 10 Accuracy روی داده‌های تست: {results[2] * 100:.2f}%")

# ذخیره مدل نهایی
model.save('hybrid_recommender_model.keras')
print("بهترین مدل با موفقیت در فایل 'hybrid_recommender_model.keras' ذخیره شد.")