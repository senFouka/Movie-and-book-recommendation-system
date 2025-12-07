# Deep Hybrid Sequential Recommender System 🧠🎬📚

This project implements a state-of-the-art **Hybrid Recommender System** that combines **Neural Collaborative Filtering (NCF)** for capturing long-term user preferences with **Dual-LSTM** and **Attention Mechanisms** for modeling short-term sequential behavior and temporal dynamics.

The system is designed to predict the next item (Movie or Book) a user is likely to interact with, achieving high accuracy on standard datasets.

---

## 🏗️ Architecture

The model utilizes a multi-input architecture:
1.  **User Branch (NCF):** Learns static user embeddings to capture general taste.
2.  **Item Sequence Branch (LSTM):** Processes the history of items viewed by the user.
3.  **Time Sequence Branch (LSTM):** Processes the temporal context (seasonality/time-of-day).
4.  **Attention Mechanism:** Dynamically weighs the importance of different parts of the history.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `build_dataset.py` | Preprocesses raw data (MovieLens/Goodbooks) into `.npz` format. |
| `train_final_hybrid_model.py` | Trains the main **Hybrid Model** (NCF + Dual-LSTM + Attention). |
| `evaluate_hybrid_ncf.py` | Evaluates the model using the standard **Hit Ratio@10 (1-vs-100)** method. |
| `run_hybrid_interactive.py` | **Interactive Demo**: Chat with the AI to get real-time recommendations. |
| `movie_data/` | Contains MovieLens datasets and trained models. |
| `book_data/` | Contains Goodbooks datasets and trained models. |

---

## 🚀 Installation

1.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃‍♂️ Usage Guide

### 1. Data Preparation
First, process the raw datasets into sequence data:
```bash
python build_dataset.py movie
python build_dataset.py book

### 2. Training the Model
Train the advanced Hybrid model. This will save the model to the respective data folder.
```bash
python train_final_hybrid_model.py movie
python train_final_hybrid_model.py book

### 3. Evaluation (The Benchmark) 📊
Run the evaluation script to calculate **Hit Ratio@10** using the standard academic method (Negative Sampling: 1 real item vs 99 negative items).

```bash
python evaluate_hybrid_ncf.py


### 4. Interactive Demo (Real-world Test) 🎮
Talk to the AI! This script loads the user history, offers recommendations, and updates based on your choices in real-time.

**For Movies:**
```bash
python run_hybrid_interactive.py movie 1
(Replace 1 with any User ID to see different histories)

For Books:
python run_hybrid_interactive.py book 10

```bash

### 4. 🏆 Results & Performance 🎮
We evaluated the model using two distinct metrics to demonstrate both real-world difficulty and academic performance:

Full Ranking: Ranking the correct item among ALL items (e.g., 1 vs 3700). This represents the "Hard Mode".

NCF Method (HR@10): Ranking the correct item among 100 items (1 positive + 99 negatives). This is the standard Academic Benchmark.

Dataset                 Model Architecture          Full Ranking Accuracy           Standard HR@10 (NCF Method)
MovieLens 1M            Hybrid (LSTM+Attn)          24.71%                          89.80% 🌟
Goodbooks-10k,Hybrid    (LSTM+Attn)                 30.07%                          90.50% 🥇
Netflix Prize,Hybrid    (Deep)                      32.56%                          N/A (Scalability Test)


> **Conclusion:** The model achieves **~90% accuracy** using the standard evaluation protocol (He et al., 2017), significantly outperforming baseline models and demonstrating robustness across different domains (Movies and Books).

---

## 📚 References

1.  **NCF:** He et al., "Neural Collaborative Filtering", WWW 2017.
2.  **SASRec:** Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018.
3.  **BERT4Rec:** Sun et al., "BERT4Rec: Sequential Recommendation with Transformers", CIKM 2019.

---
*Author: Taha Arab*