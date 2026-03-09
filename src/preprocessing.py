import pandas as pd
import numpy as np
import os

def run_preprocessing(config):
    # 1. Kaynak Dosyaların Varlığını Kontrol Et
    train_path = config["paths"]["train_path"]
    test_path = config["paths"]["test_path"]

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"HATA: 'Data' klasörü içinde trainDATA.csv veya testDATA.csv bulunamadı!")
        return

    # 2. Verileri Yükle
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)

    # 3. Temizlik ve Özellik Mühendisliği (Aynı Mantık)
    X_train = X_train.drop('name', axis=1)
    X_test = X_test.drop('name', axis=1)

    y_train = X_train['selling_price']
    X_train = X_train.drop('selling_price', axis=1)
    y_test = X_test['selling_price']
    X_test = X_test.drop('selling_price', axis=1)

    current_year = 2026
    X_train['car_age'] = current_year - X_train['year']
    X_test['car_age'] = current_year - X_test['year']
    X_train = X_train.drop('year', axis=1)
    X_test = X_test.drop('year', axis=1)

    # 4. Kategorik Değişkenler ve One-Hot Encoding
    X_train['is_train'] = 1
    X_test['is_train'] = 0
    combined_df = pd.concat([X_train, X_test], axis=0)

    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
    combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True).astype(float)

    X_train = combined_df[combined_df['is_train'] == 1].drop('is_train', axis=1)
    X_test = combined_df[combined_df['is_train'] == 0].drop('is_train', axis=1)

    # 5. Ölçeklendirme (Scaling)
    cols_to_scale = X_train.columns
    train_mean = X_train[cols_to_scale].mean()
    train_std = X_train[cols_to_scale].std()
    train_std[train_std == 0] = 1 

    X_train[cols_to_scale] = (X_train[cols_to_scale] - train_mean) / train_std
    X_test[cols_to_scale] = (X_test[cols_to_scale] - train_mean) / train_std

    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_scaled = (y_train - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    return X_train,  y_train_scaled, X_test, y_test_scaled
