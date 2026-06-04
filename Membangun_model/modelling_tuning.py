import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# Menghubungkan script ke MLflow Tracking Server lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Nama eksperimen untuk proses tuning
mlflow.set_experiment("Heart_Disease_Tuning_Experiment")


# Membaca dataset hasil preprocessing
dataset_path = os.path.join(
    "heart_disease_preprocessing",
    "heart_disease_preprocessed.csv"
)

df = pd.read_csv(dataset_path)

print("Dataset berhasil dimuat.")
print("Ukuran dataset:", df.shape)


# Memisahkan fitur dan target
X = df.drop(columns=["target"])
y = df["target"]


# Membagi dataset menjadi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Menentukan model dasar
model = RandomForestClassifier(
    random_state=42
)


# Menentukan kombinasi hyperparameter
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}


# Melakukan hyperparameter tuning
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


# Memulai MLflow run
with mlflow.start_run():

    print("Proses hyperparameter tuning dimulai...")

    grid_search.fit(
        X_train,
        y_train
    )

    best_model = grid_search.best_estimator_

    print("Hyperparameter tuning selesai.")
    print("Best parameters:", grid_search.best_params_)


    # Melakukan prediksi
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]


    # Menghitung metrik evaluasi
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba
    )


    # Manual logging hyperparameter
    mlflow.log_params(
        grid_search.best_params_
    )


    # Manual logging metrik
    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    mlflow.log_metric(
        "best_cv_score",
        grid_search.best_score_
    )


    # Menyimpan model ke artefak MLflow
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model"
    )


    # Membuat folder artefak tambahan
    os.makedirs(
        "artifacts",
        exist_ok=True
    )


    # Menyimpan confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot()

    plt.title(
        "Confusion Matrix - Heart Disease Model"
    )

    confusion_matrix_path = os.path.join(
        "artifacts",
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        bbox_inches="tight"
    )

    plt.close()


    # Menyimpan classification report
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    classification_report_path = os.path.join(
        "artifacts",
        "classification_report.json"
    )

    with open(
        classification_report_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4
        )


    # Logging artefak tambahan
    mlflow.log_artifact(
        confusion_matrix_path
    )

    mlflow.log_artifact(
        classification_report_path
    )


    print("\nHasil evaluasi model terbaik:")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", f1)
    print("ROC-AUC  :", roc_auc)
    print("\nModel dan artefak berhasil disimpan ke MLflow.")