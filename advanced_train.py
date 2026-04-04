import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading dataset...")
df = pd.read_csv("dataset_features.csv")
df = df.dropna()

X = df[["EAR", "Left_EAR", "Right_EAR", "MAR", "MOE", "Tilt"]]
y = df["Label"].map({"Normal": 0, "Sleepy": 1})

print(f"Original Dataset Size: {len(df)} samples")
print(f"Class Distribution before SMOTE:\n{df['Label'].value_counts()}")

print("Applying SMOTE for Data Balancing...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Class Distribution after SMOTE:\n{pd.Series(y_resampled).value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print("Scaling Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Advanced XGBoost Classifier...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"SUCCESS: Model Accuracy on Test Data: {accuracy * 100:.2f}%")
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Sleepy"]))

joblib.dump(model, "xgboost_drowsiness_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")
print("New advanced XGBoost model saved to 'xgboost_drowsiness_model.pkl'")
print("Scaler saved to 'feature_scaler.pkl'")

