import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import os

def get_recommendations(domain, item_history_ids, time_history_names, top_k=10):
    data_dir = f"{domain}_data"
    
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
        print(f"لطفاً ابتدا 'python train_lstm_no_attention.py {domain}' را اجرا کنید.")
        return None

    MAX_SEQUENCE_LENGTH = 20
    
    known_items = [item for item in item_history_ids if item in item_encoder.classes_]
    if not known_items:
        print("هشدار: هیچ‌کدام از آیتم‌های تاریخچه در داده‌های آموزشی یافت نشد.")
        return None
        
    item_history_encoded = item_encoder.transform(known_items)
    time_history_encoded = time_encoder.transform(time_history_names)
    
    X_item_seq = pad_sequences([item_history_encoded], maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    X_time_seq = pad_sequences([time_history_encoded], maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    
    inputs = [X_item_seq, X_time_seq]
    predictions = model.predict(inputs)
    
    top_k_indices = predictions[0].argsort()[::-1][:top_k]
    predicted_item_ids = item_encoder.inverse_transform(top_k_indices)
    
    recommendations_df = items[items['ItemID'].isin(predicted_item_ids)]
    recommendations_df['ItemID'] = pd.Categorical(recommendations_df['ItemID'], categories=predicted_item_ids, ordered=True)
    recommendations = recommendations_df.sort_values('ItemID')
    
    return recommendations

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("خطا در اجرا. لطفاً دامنه را مشخص کنید.")
        print("مثال: python predict.py movie")
        print("   یا: python predict.py book")
    else:
        domain = sys.argv[1]
        
        if domain == 'movie':
            simulated_item_history = [590, 1196, 1210] 
            simulated_time_history = ['Morning', 'Afternoon', 'Night']
        elif domain == 'book':
            simulated_item_history = [1, 2, 3] 
            simulated_time_history = ['AllDay', 'AllDay', 'AllDay']
        
        print(f"\n--- شروع پیش‌بینی برای {domain} بر اساس تاریخچه: {simulated_item_history} ---")
        
        recommendations = get_recommendations(domain, simulated_item_history, simulated_time_history, top_k=10)

        if recommendations is not None:
            print(f"\n✅ --- 10 {domain} پیشنهادی بر اساس تاریخچه ---")
            print(recommendations.to_string(index=False))