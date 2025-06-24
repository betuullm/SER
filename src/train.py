import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from sklearn.ensemble import RandomForestClassifier

def load_data(features_path, labels_path, extra_features_path=None):
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    if extra_features_path:
        extra_features = pd.read_csv(extra_features_path)
        # 'file' ve 'folder' üzerinden birleştir
        features = pd.merge(features, extra_features, on=['file', 'folder'], how='left')
    num_labels = len(labels)
    repeats = len(features) // num_labels
    remainder = len(features) % num_labels
    full_labels = list(labels["Label"].values) * repeats + list(labels["Label"].values)[:remainder]
    features["label"] = full_labels
    return features

def split_data(features, test_size=0.2, val_size=0.1, random_state=42):
    # Özellik sütunlarını açıkça seç
    X = features.drop(columns=['file', 'folder', 'label'], errors='ignore')
    print(f"Toplam kullanılan özellik sayısı: {X.shape[1]}")
    y = features["label"]
    # Önce test setini ayır
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    # Sonra kalan veriden validation setini ayır
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_relative_size, random_state=random_state, stratify=y_temp)
    return X_train.values, X_val.values, X_test.values, y_train, y_val, y_test

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    clf = RandomForestClassifier(random_state=42, n_estimators=200)
    clf.fit(X_train, y_train)
    print(f"Random Forest ile eğitim tamamlandı.")
    # Validation seti ile değerlendirme
    y_val_pred = clf.predict(X_val)
    print("\nValidation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))
    # Test seti ile değerlendirme
    y_test_pred = clf.predict(X_test)
    print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Test Classification Report:\n", classification_report(y_test, y_test_pred))
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    return clf

def main(args):
    features = load_data(args.features, args.labels, extra_features_path="..//data//train_features.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, test_size=args.test_size, val_size=args.val_size, random_state=args.random_state)
    print(f"Eğitim: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    clf = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP ile Duygu Sınıflandırma")
    parser.add_argument('--features', type=str, default="..//data//train_features_wav2vec.csv", help='Özellik dosyası yolu')
    parser.add_argument('--labels', type=str, default="..//data//Labels.csv", help='Label dosyası yolu')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test seti oranı')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation seti oranı')
    parser.add_argument('--random_state', type=int, default=42, help='Rastgelelik sabiti')
    args = parser.parse_args()
    main(args)
