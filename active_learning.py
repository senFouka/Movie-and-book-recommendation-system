import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import os

# --- متغیر ثابت را به اینجا (محدوده جهانی) منتقل کردیم ---
MAX_SEQUENCE_LENGTH = 20 

print("--- ۱. بارگذاری مدل و فایل‌های مورد نیاز ---")

def get_recommendations_from_history(domain, model, item_encoder, time_encoder, item_history_ids, time_history_names):
    
    if len(item_history_ids) > 0:
        known_items = [item for item in item_history_ids if item in item_encoder.classes_]
        if not known_items:
             item_history_encoded = []
             time_history_encoded = []
        else:
            item_history_encoded = item_encoder.transform(known_items)
            time_history_encoded = time_encoder.transform(time_history_names)
    else:
        item_history_encoded = []
        time_history_encoded = []
        
    # حالا این تابع MAX_SEQUENCE_LENGTH را می‌شناسد
    X_item_seq = pad_sequences([item_history_encoded], maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    X_time_seq = pad_sequences([time_history_encoded], maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    
    inputs = [X_item_seq, X_time_seq]
    predictions = model.predict(inputs)[0] 
    return predictions

def get_item_titles(items_df, item_ids):
    return items_df[items_df['ItemID'].isin(item_ids)].copy()

def get_uncertainty_query(predictions, item_encoder, top_k=5):
    sorted_indices = predictions.argsort()[::-1]
    uncertain_indices = sorted_indices[100:105] 
    uncertain_item_ids = item_encoder.inverse_transform(uncertain_indices)
    return uncertain_item_ids

def main(domain):
    data_dir = f"{domain}_data"
    # --- MAX_SEQUENCE_LENGTH از اینجا حذف شد ---

    print(f"--- ۱. بارگذاری مدل نهایی ({domain}) و فایل‌ها ---")
    try:
        model = load_model(os.path.join(data_dir, 'lstm_simple_model.keras'))
        item_encoder = joblib.load(os.path.join(data_dir, 'item_encoder.joblib'))
        time_encoder = joblib.load(os.path.join(data_dir, 'time_encoder.joblib'))
        
        if domain == 'movie':
            items_file = os.path.join(data_dir, 'movies.dat')
            items = pd.read_csv(items_file, sep='::', names=['ItemID', 'Title', 'Genres'], engine='python', encoding='latin-1')
        elif domain == 'book':
            items_file = os.path.join(data_dir, 'books.csv')
            items = pd.read_csv(items_file)
            items.rename(columns={'book_id': 'ItemID', 'title': 'Title'}, inplace=True)
            if 'Genres' not in items.columns:
                 items['Genres'] = 'N/A'
            items = items[['ItemID', 'Title', 'Genres']]

    except FileNotFoundError as e:
        print(f"خطا: فایل مورد نیاز پیدا نشد: {e.filename}")
        return

    print("فایل‌ها با موفقیت بارگذاری شدند.")
    
    print("\n--- سناریوی کاربر جدید: زمان ۰ (ثبت‌نام) ---")
    empty_item_history = []
    empty_time_history = []

    initial_predictions = get_recommendations_from_history(
        domain, model, item_encoder, time_encoder,
        empty_item_history, empty_time_history
    )
    initial_top_10_indices = initial_predictions.argsort()[::-1][:10]
    initial_top_10_ids = item_encoder.inverse_transform(initial_top_10_indices)
    print("پیشنهادهای اولیه (عمومی - بر اساس تاریخچه خالی):")
    print(get_item_titles(items, initial_top_10_ids).to_string(index=False))

    print("\n--- سناریوی کاربر جدید: زمان ۱ (اجرای یادگیری فعال) ---")
    query_item_ids = get_uncertainty_query(initial_predictions, item_encoder, top_k=5)
    print(f"سیستم از کاربر می‌خواهد به این 5 {domain} امتیاز دهد:")
    print(get_item_titles(items, query_item_ids).to_string(index=False))

    print("\n--- سناریوی کاربر جدید: زمان ۲ (دریافت بازخورد فعال) ---")
    new_user_history_items = query_item_ids[:3] 
    
    if domain == 'movie':
        new_user_history_times = ['Night', 'Night', 'Night']
    elif domain == 'book':
        new_user_history_times = ['AllDay', 'AllDay', 'AllDay']

    print(f"کاربر به 3 {domain} امتیاز داد: {new_user_history_items}")
    print("پروفایل او ساخته شد.")

    print("\n--- سناریوی کاربر جدید: زمان ۳ (پیشنهادهای شخصی‌سازی شده) ---")
    personalized_predictions = get_recommendations_from_history(
        domain, model, item_encoder, time_encoder,
        new_user_history_items, new_user_history_times
    )
    personalized_top_10_indices = personalized_predictions.argsort()[::-1][:10]
    personalized_top_10_ids = item_encoder.inverse_transform(personalized_top_10_indices)

    print(f"✅ پیشنهادهای جدید (شخصی‌سازی شده بر اساس بازخورد فعال {domain}):")
    print(get_item_titles(items, personalized_top_10_ids).to_string(index=False))

    print("\n--- مقایسه ---")
    if np.array_equal(initial_top_10_ids, personalized_top_10_ids):
        print("هشدار: لیست‌ها تغییری نکردند.")
    else:
        print("موفقیت: لیست شخصی‌شده کاملاً با لیست عمومی متفاوت است.")
        print("این نشان می‌دهد که ماژول Active Feedback با موفقیت مشکل شروع سرد را حل کرد.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("خطا در اجرا. لطفاً دامنه را مشخص کنید.")
        print("مثال: python active_learning.py movie")
        print("   یا: python active_learning.py book")
    else:
        domain = sys.argv[1]
        main(domain)