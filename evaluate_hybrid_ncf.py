import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import random
from tqdm import tqdm

def evaluate_hr_at_10_dual(model, X_item, X_time, y, n_items, num_negatives=99):
    hits = 0
    total = len(y)
    
    # تست روی ۱۰۰۰ نمونه تصادفی
    sample_indices = random.sample(range(total), min(1000, total)) 
    
    print(f"--- شروع ارزیابی مدل Dual-LSTM به روش NCF (1 real vs {num_negatives} negatives) ---")
    
    for idx in tqdm(sample_indices):
        # 1. داده‌های ورودی واقعی
        user_item_seq = X_item[idx]
        user_time_seq = X_time[idx]
        target_item = y[idx] 
        
        # 2. ساختن 99 تا آیتم منفی
        negatives = []
        while len(negatives) < num_negatives:
            neg_item = random.randint(1, n_items - 1)
            if neg_item != target_item:
                negatives.append(neg_item)
        
        items_to_rank = [target_item] + negatives
        
        # 3. پیش‌بینی (مخصوص مدل Dual-LSTM با 2 ورودی)
        pred_vector = model.predict([
            np.array([user_item_seq]), 
            np.array([user_time_seq])
        ], verbose=0)[0]
        
        # 4. امتیازدهی
        scores = {}
        for item_id in items_to_rank:
            if item_id < len(pred_vector):
                scores[item_id] = pred_vector[item_id]
            else:
                scores[item_id] = -1.0
            
        # 5. رتبه‌بندی
        ranked_items = sorted(scores, key=scores.get, reverse=True)
        top_10 = ranked_items[:10]
        
        if target_item in top_10:
            hits += 1
            
    return hits / len(sample_indices)

def main():
    domain = 'book' 
    data_dir = f"{domain}_data"
    processed_file = os.path.join(data_dir, 'processed_data.npz')
    
    # --- اصلاح نام فایل: اشاره به مدلی که واقعاً دارید ---
    model_file = os.path.join(data_dir, 'lstm_dual_masked_attention_model.keras')

    print(f"بارگذاری داده‌های {domain}...")
    data = np.load(processed_file)
    
    limit = 20000
    X_item_test = data['X_item'][-limit:]
    X_time_test = data['X_time'][-limit:]
    y_test = data['y'][-limit:]
    n_items = int(data['n_items'])
    
    print(f"بارگذاری مدل از {model_file}...")
    try:
        model = load_model(model_file)
    except Exception as e:
        print(f"خطا: {e}")
        return
    
    # محاسبه
    hr_10 = evaluate_hr_at_10_dual(model, X_item_test, X_time_test, y_test, n_items)
    
    print("\n" + "="*50)
    print(f"✅ DUAL-LSTM BOOK HIT RATIO @ 10 (NCF Method): {hr_10 * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()