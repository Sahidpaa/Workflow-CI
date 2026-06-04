import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Menghubungkan script ke MLflow Tracking Server lokal
mlflow.set_tracking_uri("file:./mlruns")

# Nama eksperimen yang akan muncul di dashboard MLflow
mlflow.set_experiment("Heart_Disease_Experiment")


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


# Membagi data menjadi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Mengaktifkan automatic logging MLflow
mlflow.sklearn.autolog()


# Memulai training model
with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    print("Training selesai.")
    print("Accuracy:", accuracy)