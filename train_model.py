"""
Bookshop Management System - Model Training Script
Trains:
1. Inventory Health & Restock Classifier (Random Forest Classifier)
2. Bookshop Revenue & Valuation Estimator (Random Forest Regressor)
3. Content-Based Book Recommendation Engine (TF-IDF + Cosine Similarity)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_absolute_error

def train_bookshop_models(data_path='books_scraped.csv', output_dir='models'):
    print("Loading books scraped dataset...")
    df = pd.read_csv(data_path)
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Feature Engineering & Cleaning
    # ---------------------------------------------------------
    rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
    df['rating_num'] = df['Star_rating'].map(rating_map).fillna(3).astype(int)
    df['inventory_value'] = (df['Price'] * df['Quantity']).round(2)

    # Actionable Inventory Status:
    # - Urgent Restock: popular books (>= 3 stars) with low stock (<= 5)
    # - Monitor Stock: low rated books (< 3 stars) with low stock (<= 5)
    # - Overstocked / Clearance: books with excessive quantity (> 15)
    # - Optimal Stock: healthy mid-range stock (6 to 15 copies)
    def assign_inventory_status(row):
        if row['Quantity'] <= 5 and row['rating_num'] >= 3:
            return 'Urgent Restock'
        elif row['Quantity'] <= 5:
            return 'Monitor Stock'
        elif row['Quantity'] > 15:
            return 'Overstocked / Clearance'
        else:
            return 'Optimal Stock'

    df['inventory_status'] = df.apply(assign_inventory_status, axis=1)

    print(f"Total Books: {len(df)}")
    print(f"Inventory Status Distribution:\n{df['inventory_status'].value_counts()}")

    # ---------------------------------------------------------
    # 2. Train Inventory Restock Classifier
    # ---------------------------------------------------------
    print("\n--- Training Inventory Restock Classifier ---")
    # One-hot encode category for machine learning
    top_categories = df['Book_category'].value_counts().head(20).index.tolist()
    df['category_grouped'] = df['Book_category'].apply(lambda c: c if c in top_categories else 'Other')
    cat_dummies = pd.get_dummies(df['category_grouped'], prefix='cat', drop_first=True)

    clf_X = pd.concat([
        df[['Price', 'Quantity', 'rating_num']],
        cat_dummies
    ], axis=1)
    clf_y = df['inventory_status']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        clf_X, clf_y, test_size=0.2, random_state=42, stratify=clf_y
    )

    clf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf_model.fit(X_train_c, y_train_c)
    clf_preds = clf_model.predict(X_test_c)

    acc = accuracy_score(y_test_c, clf_preds)
    print(f"Inventory Classifier Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test_c, clf_preds))

    # ---------------------------------------------------------
    # 3. Train Inventory Valuation Regressor
    # ---------------------------------------------------------
    print("\n--- Training Inventory Valuation Regressor ---")
    reg_X = pd.concat([df[['Price', 'rating_num']], cat_dummies], axis=1)
    reg_y = df['inventory_value']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        reg_X, reg_y, test_size=0.2, random_state=42
    )
    reg_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    reg_model.fit(X_train_r, y_train_r)
    reg_preds = reg_model.predict(X_test_r)

    r2 = r2_score(y_test_r, reg_preds)
    mae = mean_absolute_error(y_test_r, reg_preds)
    print(f"Valuation Regressor R²: {r2:.4f}, MAE: £{mae:.2f}")

    # ---------------------------------------------------------
    # 4. Content-Based Recommendation Engine
    # ---------------------------------------------------------
    print("\n--- Building Book Recommendation Engine ---")
    # Combine title semantics and category tokens
    df['recommender_corpus'] = df['Title'] + " " + df['Book_category'] + " " + df['Book_category']
    tfidf = TfidfVectorizer(stop_words='english', max_features=1500)
    tfidf_matrix = tfidf.fit_transform(df['recommender_corpus'])

    # ---------------------------------------------------------
    # 5. Save Model Artifacts
    # ---------------------------------------------------------
    joblib.dump(clf_model, os.path.join(output_dir, 'restock_classifier.pkl'))
    joblib.dump(reg_model, os.path.join(output_dir, 'pricing_model.pkl'))
    joblib.dump({
        'tfidf_vectorizer': tfidf,
        'tfidf_matrix': tfidf_matrix,
        'features': clf_X.columns.tolist()
    }, os.path.join(output_dir, 'book_recommender.pkl'))

    metadata = {
        'total_books': len(df),
        'classifier_accuracy': round(float(acc), 4),
        'categories': sorted(df['Book_category'].unique().tolist()),
        'top_categories': top_categories,
        'feature_columns': clf_X.columns.tolist()
    }
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    df.to_csv(os.path.join(output_dir, 'books_enriched.csv'), index=False)
    print(f"All models and artifacts successfully saved to '{output_dir}/'")

if __name__ == '__main__':
    train_bookshop_models()
