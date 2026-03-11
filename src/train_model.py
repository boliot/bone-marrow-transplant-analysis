import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing import preprocess_pipeline
def get_models():
    return {"Random Forest": RandomForestClassifier(n_estimators=100,class_weight="balanced",random_state=42),"XGBoost": XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            verbose=-1
        ),
    }

def evaluate_model(model, X_test, y_test) -> dict:
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "F1-Score":  round(f1_score(y_test, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
    }


def train_all_models(X_train, X_test, y_train, y_test):
    models = get_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n{'='*45}")
        print(f"  Training : {name}")
        print(f"{'='*45}")

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        trained_models[name] = model

        for metric, value in metrics.items():
            print(f"  {metric:<12}: {value}")

    print("\n\n📊 MODEL COMPARISON TABLE")
    print("─" * 65)
    df_results = pd.DataFrame(results).T
    print(df_results.to_string())
    print("─" * 65)

    best_name = df_results["ROC-AUC"].idxmax()
    best_model = trained_models[best_name]
    print(f"\n✅ Best model: {best_name}  (ROC-AUC = {df_results.loc[best_name, 'ROC-AUC']})")
    print(f"\n   Reasoning: In a medical context, ROC-AUC is the most reliable")
    print(f"   metric because it measures discrimination regardless of threshold.")
    print(f"   High Recall is also critical to avoid missing transplant failures.")

    return best_model, best_name, results, trained_models

def save_model(model, model_name: str, output_dir: str = "models"):
    """Save the trained model as a .pkl file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = model_name.lower().replace(" ", "_") + ".pkl"
    path = os.path.join(output_dir, filename)
    joblib.dump(model, path)
    print(f"\n💾 Model saved → {path}")
    return path

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/bone-marrow.arff"

    print("🔄 Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_pipeline(data_path)

    print("\n🤖 Training models...")
    best_model, best_name, results, all_models = train_all_models(
        X_train, X_test, y_train, y_test
    )

    save_model(best_model, best_name)
    print("\n✅ Training complete!")