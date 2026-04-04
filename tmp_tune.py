import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.neighbors import SVC

def run_tests():
    df = pd.read_csv('dataset_features.csv').dropna()
    X = df[["EAR", "Left_EAR", "Right_EAR", "MAR", "MOE", "Tilt"]]
    y = df["Label"].map({"Normal": 0, "Sleepy": 1})

    # Smote
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "XGBoost Default": XGBClassifier(random_state=42),
        "XGBoost Tuned": XGBClassifier(n_estimators=1000, max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42),
        "Random Forest Tuned": RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=500, random_state=42),
        "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=500, max_depth=10, random_state=42)
    }

    for name, model in models.items():
        try:
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
            acc = accuracy_score(y_test, pred)
            print(f"{name}: {acc*100:.2f}%")
        except Exception as e:
            print(f"{name} Failed: {e}")

if __name__ == "__main__":
    run_tests()
