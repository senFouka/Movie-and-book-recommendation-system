import matplotlib.pyplot as plt
import numpy as np

def plot_comparison():
    # داده‌ها
    models = ['Baseline (PopRec)', 'SASRec (2018)', 'BERT4Rec (2019)', 'My Model (Hybrid)']
    
    # 1. نمودار مقایسه با مقالات (MovieLens) - معیار HR@10
    # اعداد مقالات تقریبی و بر اساس گزارش‌های معتبر هستند
    scores_movie = [14.3, 25.5, 28.1, 24.71] 
    colors = ['gray', 'blue', 'orange', 'green']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores_movie, color=colors, alpha=0.8)
    
    # اضافه کردن عدد روی نمودار
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval}%', ha='center', fontweight='bold')

    plt.title('Comparison with SOTA Models (MovieLens 1M - Full Ranking)', fontsize=14)
    plt.ylabel('Hit Ratio @ 10 (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('chart_comparison_movies.png')
    print("✅ نمودار 1 ساخته شد: chart_comparison_movies.png")

    # 2. نمودار معجزه تغییر معیار (Full Ranking vs NCF Method)
    labels = ['Full Ranking (Hard)', 'NCF Method (Standard)']
    movie_scores = [24.71, 89.80]
    book_scores = [30.07, 90.50]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, movie_scores, width, label='Movies', color='blue')
    plt.bar(x + width/2, book_scores, width, label='Books', color='green')

    plt.ylabel('Accuracy (%)')
    plt.title('Impact of Evaluation Metric on Accuracy', fontsize=14)
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # اضافه کردن اعداد
    for i in range(len(movie_scores)):
        plt.text(i - width/2, movie_scores[i] + 2, f'{movie_scores[i]}%', ha='center', color='black', fontweight='bold')
        plt.text(i + width/2, book_scores[i] + 2, f'{book_scores[i]}%', ha='center', color='black', fontweight='bold')

    plt.savefig('chart_metric_impact.png')
    print("✅ نمودار 2 ساخته شد: chart_metric_impact.png")

if __name__ == "__main__":
    plot_comparison()