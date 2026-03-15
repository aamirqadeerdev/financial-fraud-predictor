
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os


def load_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total transactions: {len(df)}")
    print(f"Fraud cases: {df['Class'].sum()}")
    print(f"Legitimate cases: {(df['Class'] == 0).sum()}")
    print(f"Fraud percentage: {df['Class'].mean() * 100:.2f}%")
    
    return df


def preprocess_data(df):
    print("\nPreprocessing data...")
    
    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Drop Time column — not useful for prediction
    X = X.drop(columns=['Time'])
    
    # Scale Amount column separately
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Save scaler for use in API
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

    
def train_model(X_train, y_train):
    print("\nTraining Random Forest model...")
    print("This may take 2-3 minutes...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    return model

def evaluate_model(model, X_test, y_test, feature_names):
    print("\nEvaluating model performance...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Print evaluation metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
          target_names=['Legitimate', 'Fraud']))
    
    # AUC-ROC Score
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"AUC-ROC Score: {auc_score:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"True Negatives  (Legitimate correctly identified): {cm[0][0]}")
    print(f"False Positives (Legitimate wrongly flagged):      {cm[0][1]}")
    print(f"False Negatives (Fraud missed):                    {cm[1][0]}")
    print(f"True Positives  (Fraud correctly caught):          {cm[1][1]}")
    
    # Feature Importance Top 10
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(importance.head(10).to_string(index=False))
    
    return auc_score


def main():
    print("=" * 60)
    print("FINANCIAL FRAUD DETECTION — MODEL TRAINING")
    print("=" * 60)
    
    # Step 1 — Load data
    df = load_data('data/creditcard.csv')
    
    # Step 2 — Preprocess data
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # Step 3 — Train model
    model = train_model(X_train, y_train)
    
    # Step 4 — Evaluate model
    auc_score = evaluate_model(model, X_test, y_test, feature_names)
    
    # Step 5 — Save model
    joblib.dump(model, 'models/model.pkl')
    print(f"\nModel saved to models/model.pkl")
    
    # Step 6 — Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"AUC-ROC Score : {auc_score:.4f}")
    print(f"Model saved   : models/model.pkl")
    print(f"Scaler saved  : models/scaler.pkl")
    print("Ready for FastAPI deployment!")
    print("=" * 60)


if __name__ == "__main__":
    main()







