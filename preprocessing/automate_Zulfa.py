import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_path):
    # Membaca dataset mentah
    df = pd.read_csv(input_path)

    print("Dataset berhasil dimuat.")
    print("Ukuran awal dataset:", df.shape)

    # Mengubah target menjadi klasifikasi biner
    # 0 = tidak terindikasi penyakit jantung
    # 1 = terindikasi penyakit jantung
    df["target"] = (df["num"] > 0).astype(int)

    # Menghapus kolom target asli
    df = df.drop(columns=["num"])

    # Menghapus data duplikat
    jumlah_data_sebelum = len(df)
    df = df.drop_duplicates()
    jumlah_data_sesudah = len(df)

    print(
        "Jumlah data duplikat yang dihapus:",
        jumlah_data_sebelum - jumlah_data_sesudah
    )

    # Menangani missing value menggunakan median
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)

    # Standardisasi fitur numerik kontinu
    numerical_features = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]

    scaler = StandardScaler()

    df[numerical_features] = scaler.fit_transform(
        df[numerical_features]
    )

    # Membuat folder output jika belum tersedia
    output_folder = os.path.dirname(output_path)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # Menyimpan dataset hasil preprocessing
    df.to_csv(output_path, index=False)

    print("Preprocessing berhasil dilakukan.")
    print("Ukuran akhir dataset:", df.shape)
    print("Total missing value:", df.isnull().sum().sum())
    print("Jumlah data duplikat:", df.duplicated().sum())
    print("Dataset tersimpan di:", output_path)


if __name__ == "__main__":
    input_file = "../heart_disease_raw/heart_disease_raw.csv"

    output_file = (
        "heart_disease_preprocessing/"
        "heart_disease_preprocessed.csv"
    )

    preprocess_data(
        input_path=input_file,
        output_path=output_file
    )