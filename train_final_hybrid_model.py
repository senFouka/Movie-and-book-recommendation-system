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
import sys 
import os 

def main(domain):
    data_dir = f"{domain}_data"
    processed_file = os.path.join(data_dir, 'processed_data.npz')
    # --- نام مدل نهایی ---
    model_file = os.path.join(data_dir, 'final_hybrid_model.keras')

    print(f"--- ۱. بارگذاری داده‌های پردازش شده از: {processed_file} ---")

    try:
        data = np.load(processed_file)
    except FileNotFoundError:
        print(f"فایل '{processed_file}' پیدا نشد.")
        return

    # --- ورودی‌ها رو کامل بارگذاری می‌کنیم ---
    X_user = data['X_user']
    X_item = data['X_item']
    X_time = data['X_time']
    y = data['y']

    n_users = int(data['n_users'])
    n_items = int(data['n_items'])
    n_time_features = int(data['n_time_features'])
    MAX_SEQUENCE_LENGTH = X_item.shape[1] 

    print(f"تعداد کاربران: {n_users}")
    print(f"تعداد آیتم‌ها: {n_items}")
    print(f"تعداد دسته‌های زمانی: {n_time_features}")

    print("\n--- ۲. تقسیم داده‌ها ---")
    indices = np.arange(y.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

    # --- مدل نهایی ما ۳ ورودی دارد ---
    X_train = [X_user[train_indices], X_item[train_indices], X_time[train_indices]]
    X_test = [X_user[test_indices], X_item[test_indices], X_time[test_indices]]

    y_train = y[train_indices]
    y_test = y[test_indices]
    
    print(f"تعداد نمونه‌های آموزشی: {len(y_train)}")


    print("\n--- ۳. ساخت معماری نهایی (NCF + Dual-LSTM + Masked-Attention) ---")
    
    USER_EMBEDDING_SIZE = 64
    ITEM_EMBEDDING_SIZE = 64
    TIME_EMBEDDING_SIZE = 16
    LSTM_UNITS_ITEM = 64
    LSTM_UNITS_TIME = 16

    # --- تعریف ۳ ورودی ---
    user_input = Input(shape=(1,), name='UserInput')
    item_seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='ItemSequenceInput')
    time_seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='TimeSequenceInput')

    # --- شاخه ۱: NCF (سلیقه کلی کاربر) ---
    user_embedding_layer = Embedding(input_dim=n_users, output_dim=USER_EMBEDDING_SIZE, name='UserEmbedding')
    user_vec = Flatten(name='FlattenUser')(user_embedding_layer(user_input))

    # --- شاخه ۲: رفتار ترتیبی (LSTM + Time + Attention) ---
    # بخش آیتم
    item_embedding_layer = Embedding(input_dim=n_items, output_dim=ITEM_EMBEDDING_SIZE, name='ItemEmbedding', mask_zero=True)
    item_seq_embedding = item_embedding_layer(item_seq_input)
    lstm_item_out = LSTM(LSTM_UNITS_ITEM, return_sequences=True, name='LSTM_Item')(item_seq_embedding)

    # بخش زمان
    time_embedding_layer = Embedding(input_dim=n_time_features, output_dim=TIME_EMBEDDING_SIZE, name='TimeEmbedding', mask_zero=True)
    time_seq_embedding = time_embedding_layer(time_seq_input)
    lstm_time_out = LSTM(LSTM_UNITS_TIME, return_sequences=True, name='LSTM_Time')(time_seq_embedding)

    # ادغام خروجی‌های LSTM (ماسک‌ها حفظ می‌شوند)
    combined_lstm_out = Concatenate(axis=2, name='CombineLSTMs')([lstm_item_out, lstm_time_out])

    # اعمال Attention روی داده‌های تمیز
    attention_out = Attention(name='Attention')([combined_lstm_out, combined_lstm_out])
    # خلاصه کردن خروجی Attention به یک بردار
    context_vec = GlobalAveragePooling1D(name='AttentionPooling')(attention_out)
    
    # --- ادغام نهایی (کلیدی‌ترین بخش) ---
    # ترکیب سلیقه کلی (NCF) + رفتار اخیر (LSTM+Attention)
    final_combined_vec = Concatenate(name='Combine_NCF_LSTM')([user_vec, context_vec])

    # --- شبکه MLP نهایی ---
    dense_1 = Dense(128, activation='relu', name='Dense_1')(final_combined_vec)
    dense_2 = Dense(64, activation='relu', name='Dense_2')(dense_1)
    output = Dense(n_items, activation='softmax', name='Output')(dense_2)

    # --- ساخت مدل نهایی ---
    model = Model(
        inputs=[user_input, item_seq_input, time_seq_input], 
        outputs=output,
        name='Final_Hybrid_Model'
    )
    model.summary()

    print("\n--- ۴. کامپایل و آموزش کامل مدل ---")
    metrics = ['sparse_categorical_accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc')]
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=metrics)
    early_stopper = EarlyStopping(monitor='val_top_10_acc', patience=3, verbose=1, mode='max', restore_best_weights=True)

    print(f"شروع آموزش مدل نهایی {domain}...")
    model.fit(X_train, y_train, batch_size=256, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopper])

    print("\n--- ۵. ارزیابی نهایی مدل ---")
    results = model.evaluate(X_test, y_test)
    print(f"✅ مدل {domain} (Final Hybrid) با موفقیت آموزش دید.")
    print(f"Loss (خطا) روی داده‌های تست: {results[0]:.4f}")
    print(f"Accuracy (دقت) روی داده‌های تست: {results[1] * 100:.2f}%")
    print(f"Top 10 Accuracy روی داده‌های تست: {results[2] * 100:.2f}%")

    model.save(model_file)
    print(f"مدل نهایی در '{model_file}' ذخیره شد.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("خطا در اجرا. لطفاً دامنه را مشخص کنید.")
        print("مثال: python train_final_hybrid_model.py movie")
        print("   یا: python train_final_hybrid_model.py book")
    else:
        domain = sys.argv[1]
        main(domain)