import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import sys # <-- جدید
import os # <-- جدید

def main(domain):
    data_dir = f"{domain}_data"
    processed_file = os.path.join(data_dir, 'processed_data.npz')
    model_file = os.path.join(data_dir, 'lstm_simple_model.keras')

    print(f"--- ۱. بارگذاری داده‌های پردازش شده از: {processed_file} ---")

    try:
        data = np.load(processed_file)
    except FileNotFoundError:
        print(f"فایل '{processed_file}' پیدا نشد.")
        print(f"لطفاً ابتدا 'python build_dataset.py {domain}' را اجرا کنید.")
        return

    # --- تغییر: X_movie به X_item تغییر نام یافت ---
    X_item = data['X_item'] 
    X_time = data['X_time']
    y = data['y']

    n_items = int(data['n_items'])
    n_time_features = int(data['n_time_features'])
    MAX_SEQUENCE_LENGTH = X_item.shape[1] 

    print(f"تعداد آیتم‌ها: {n_items}")
    print(f"تعداد دسته‌های زمانی: {n_time_features}")
    print(f"حداکثر طول دنباله: {MAX_SEQUENCE_LENGTH}")

    print("\n--- ۲. تقسیم داده‌ها ---")
    indices = np.arange(y.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

    X_train = [X_item[train_indices], X_time[train_indices]]
    X_test = [X_item[test_indices], X_time[test_indices]]

    y_train = y[train_indices]
    y_test = y[test_indices]
    
    print(f"تعداد نمونه‌های آموزشی: {len(y_train)}")
    print(f"تعداد نمونه‌های آزمایشی: {len(y_test)}")


    print("\n--- ۳. ساخت معماری مدل (LSTM ساده) ---")
    ITEM_EMBEDDING_SIZE = 64
    TIME_EMBEDDING_SIZE = 16
    LSTM_UNITS = 64

    # --- تغییر: نام ورودی‌ها عمومی شد ---
    item_seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='ItemSequenceInput')
    time_seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='TimeSequenceInput')

    item_embedding_layer = Embedding(input_dim=n_items, output_dim=ITEM_EMBEDDING_SIZE, name='ItemEmbedding')
    item_seq_embedding = item_embedding_layer(item_seq_input)

    time_embedding_layer = Embedding(input_dim=n_time_features, output_dim=TIME_EMBEDDING_SIZE, name='TimeEmbedding')
    time_seq_embedding = time_embedding_layer(time_seq_input)

    combined_seq_embedding = Concatenate(axis=2, name='CombineItemTime')([item_seq_embedding, time_seq_embedding])
    
    lstm_out_vector = LSTM(LSTM_UNITS, return_sequences=False, name='LSTM')(combined_seq_embedding)
    
    final_vec = lstm_out_vector 

    dense_1 = Dense(128, activation='relu', name='Dense_1')(final_vec)
    dense_2 = Dense(64, activation='relu', name='Dense_2')(dense_1)
    output = Dense(n_items, activation='softmax', name='Output')(dense_2) # n_items

    model = Model(
        inputs=[item_seq_input, time_seq_input], 
        outputs=output,
        name=f'LSTM_Simple_Model_{domain}'
    )
    model.summary()

    print("\n--- ۴. کامپایل و آموزش کامل مدل ---")
    metrics = ['sparse_categorical_accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc')]
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=metrics)
    early_stopper = EarlyStopping(monitor='val_top_10_acc', patience=3, verbose=1, mode='max', restore_best_weights=True)

    print(f"شروع آموزش مدل {domain}...")
    model.fit(X_train, y_train, batch_size=256, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopper])

    print("\n--- ۵. ارزیابی نهایی مدل ---")
    results = model.evaluate(X_test, y_test)
    print(f"✅ مدل {domain} با موفقیت آموزش دید.")
    print(f"Top 10 Accuracy: {results[2] * 100:.2f}%")

    model.save(model_file)
    print(f"مدل نهایی در '{model_file}' ذخیره شد.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("خطا در اجرا. لطفاً دامنه را مشخص کنید.")
        print("مثال: python train_lstm_no_attention.py movie")
        print("   یا: python train_lstm_no_attention.py book")
    else:
        domain = sys.argv[1]
        main(domain)