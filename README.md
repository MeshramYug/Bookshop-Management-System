# Bookshop Management System

An interactive bookstore inventory control, demand restock prediction, and customer book recommendation system built with Python, Scikit-Learn, Plotly, and Streamlit.

---

## 📌 Project Overview

The **Bookshop Management System** empowers independent booksellers, store managers, and staff to maintain optimal book inventory, identify high-demand books at risk of running out of stock, flag overstocked books for promotional discounts, and deliver content-based book recommendations to customers.

The system utilizes machine learning classification for inventory health scoring and Natural Language Processing (TF-IDF + Cosine Similarity) for semantic book matching.

---

## ✨ Key Features

- **Bookshop Overview Dashboard**: Real-time inventory statistics across 1,000 book titles, total copies on hand, inventory valuation (£), category breakdowns, and star rating distributions.
- **Restock Urgency & Demand Predictor**: Machine learning classifier predicting inventory health across 4 distinct operational categories:
  - `Urgent Restock`: High-rated books ($\ge 3$ stars) with low stock ($\le 5$ copies) needing immediate replenishment.
  - `Optimal Stock`: Well-balanced shelf quantity ($6 - 15$ copies).
  - `Monitor Stock`: Low stock ($\le 5$ copies) with low rating ($< 3$ stars) to observe before ordering.
  - `Overstocked / Clearance`: Excess stock ($> 15$ copies) suitable for promotional discounting.
- **Customer Book Recommendation Engine**: Content-based recommendation model matching titles based on genre semantics, title tokens, rating affinity, and stock availability.
- **Interactive Catalog Explorer**: Search and filter books by category, star rating, stock status, and price, with CSV download capability.
- **Model Performance & Analytics**: Transparent feature importances, pricing distribution, and classifier evaluation.

---

## 🧠 Machine Learning & Recommendation Architecture

| Component | Method | Objective | Metrics |
| :--- | :--- | :--- | :--- |
| **Inventory Health Classifier** | Random Forest Classifier | Categorize stock status into 4 operational tiers | **100% Accuracy**, 1.00 Macro F1 |
| **Customer Recommender** | TF-IDF + Cosine Similarity | Identify semantically similar books | 1,500 n-gram vocabulary |
| **Valuation Estimator** | Random Forest Regressor | Predict revenue potential & price affinity | Price and category sensitivity |

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ (tested on Python 3.11)
- `pip` package manager

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone <your-repository-url>
cd Mohnish
pip install -r requirements.txt
```

### 3. Model Training (Optional)
Pre-trained model artifacts are included in the `models/` directory. To retrain the models from `books_scraped.csv`:
```bash
python train_model.py
```

### 4. Running the Application
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 🌐 Streamlit Cloud Deployment Guide

1. Push this directory (`Mohnish/`) to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your repository and branch (`main`).
5. Set **Main file path** to `app.py`.
6. Click **"Deploy!"**.

---

## 📁 Repository Structure

```text
├── app.py                      # Main interactive Streamlit application
├── train_model.py              # ML classifier & recommendation training script
├── requirements.txt            # Python dependencies for deployment
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── books_scraped.csv           # Raw bookshop inventory dataset
└── models/                     # Saved model artifacts
    ├── restock_classifier.pkl  # Trained Random Forest Classifier
    ├── pricing_model.pkl       # Trained valuation regressor
    ├── book_recommender.pkl    # Serialized TF-IDF vectorizer & similarity matrix
    ├── model_metadata.json     # Category list & performance metadata
    └── books_enriched.csv      # Enriched dataset with inventory statuses
```
