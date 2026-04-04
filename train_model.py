import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")
df = pd.read_csv("dataset_features.csv")

# Ensure no empty rows
df = df.dropna()

print(f"Dataset has {len(df)} samples.")

X = df[["EAR", "Left_EAR", "Right_EAR", "MAR", "MOE", "Tilt"]]
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
print("Training Advanced RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

# Save
joblib.dump(model, "drowsiness_model.pkl")
print("New advanced model saved to drowsiness_model.pkl ✅")
