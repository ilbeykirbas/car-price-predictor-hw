import pandas as pd
import numpy as np
import os

def run_preprocessing():
    # 1. Klasör Yollarını Ayarla
    # Bu dosyanın (preprocessing.py) bulunduğu klasörden bir üst klasöre (root) çık ve Data'ya gir
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, 'Data')
    output_dir = os.path.join(data_dir, 'PreprocessedData')

    # Dosya yolları
    train_path = os.path.join(data_dir, 'trainDATA.csv')
    test_path = os.path.join(data_dir, 'testDATA.csv')

    print(f"Veriler okunuyor: {train_path}")

    # 2. Verileri yükle
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 3. Temel Temizlik (İsim ve Hedef Değişken)
    train_df = train_df.drop('name', axis=1)
    test_df = test_df.drop('name', axis=1)

    y_train = train_df['selling_price']
    train_df = train_df.drop('selling_price', axis=1)

    y_test = test_df['selling_price']
    test_df = test_df.drop('selling_price', axis=1)

    # 4. Özellik Mühendisliği (Yıl -> Yaş)
    current_year = 2026
    train_df['car_age'] = current_year - train_df['year']
    test_df['car_age'] = current_year - test_df['year']
    train_df = train_df.drop('year', axis=1)
    test_df = test_df.drop('year', axis=1)

    # 5. Kategorik Değişkenler ve One-Hot Encoding
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    combined_df = pd.concat([train_df, test_df], axis=0)

    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
    # .astype(float) ekledik ki matematiksel işlemlerde hata vermesin
    combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True).astype(float)

    # Tekrar ayır
    train_df = combined_df[combined_df['is_train'] == 1].drop('is_train', axis=1)
    test_df = combined_df[combined_df['is_train'] == 0].drop('is_train', axis=1)

    # 6. Ölçeklendirme (Scaling)
    cols_to_scale = train_df.columns
    train_mean = train_df[cols_to_scale].mean()
    train_std = train_df[cols_to_scale].std()
    train_std[train_std == 0] = 1 # Sıfıra bölme hatasını önle

    train_df[cols_to_scale] = (train_df[cols_to_scale] - train_mean) / train_std
    test_df[cols_to_scale] = (test_df[cols_to_scale] - train_mean) / train_std

    # Hedef değişken ölçeklendirme
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_scaled = (y_train - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    # 7. Kaydetme (Data/PreprocessedData klasörüne)
    train_df.to_csv(os.path.join(output_dir, 'X_train_processed.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'X_test_processed.csv'), index=False)
    y_train_scaled.to_csv(os.path.join(output_dir, 'y_train_processed.csv'), index=False)
    y_test_scaled.to_csv(os.path.join(output_dir, 'y_test_processed.csv'), index=False)

    print(f"İşlem başarıyla tamamlandı. Dosyalar {output_dir} klasörüne kaydedildi.")

# Eğer bu dosya doğrudan çalıştırılırsa fonksiyonu tetikle
if __name__ == "__main__":
    run_preprocessing()