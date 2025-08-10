import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocess import preprocess_csv

# Paths
CSV_PATH = "labeled_data.csv"
MODEL_DIR = "models/saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_candle_model.json")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Ensure save directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load and preprocess data
X, y, label_encoder = preprocess_csv(CSV_PATH, n_candles=3)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost classifier
model = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=len(label_encoder.classes_),
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and label encoder
model.save_model(MODEL_PATH)
joblib.dump(label_encoder, ENCODER_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Label encoder saved to {ENCODER_PATH}")
