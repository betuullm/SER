import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Her kayda sırayla Labels.csv'den etiket ata
features = pd.read_csv("..//data//features_wav2vec.csv")
labels = pd.read_csv("..//data//Labels.csv")
# Labels.csv'deki 172 etiketi, toplam kayıt sayısına ulaşana kadar döngüyle tekrar et
num_labels = len(labels)
repeats = len(features) // num_labels
remainder = len(features) % num_labels
full_labels = list(labels["Label"].values) * repeats + list(labels["Label"].values)[:remainder]
features["label"] = full_labels
   
# Sadece d01-ikinci paradigma-d ve d01_birinci_paradigma verilerini al
filtered_features = features[features['folder'].isin(['d01-ikinci paradigma-d', 'd01_birinci_paradigma','d02-birinci paradigma-d','d02-ikinci paradigma-d','d03-birinci paradigma-d','d03-ikinci paradigma-d','d04-birinci paradigma-d','d04-ikinci paradigma-d','d05-birinci paradigma-d','d05-ikinci paradigma-d'])]

# Sadece etiketleri 1 ve 2 olan verileri seç
filtered_features = filtered_features[filtered_features['label'].isin([1,2,3,4])]

# Özellik sütunlarını seç (son 3 sütun: file, folder, label hariç)
X = features.iloc[:, :-3].values
y = features["label"].values.astype(int)

# Özellikleri ölçekle
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 80-20 oranında train ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Eğitim veri seti boyutu: {X_train.shape[0]}")
print(f"Test veri seti boyutu: {X_test.shape[0]}")
print(f"Toplam veri: {filtered_features.shape[0]} kayıt, {X.shape[1]} özellik")

# mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', alpha=0.001, learning_rate='constant', max_iter=200, random_state=42)

# # 5-fold cross validation (sadece eğitim verisi ile)
# cv_scores = cross_val_score(mlp, X_train, y_train, cv=5)
# print(f"\nMLPClassifier 5-Fold CV Accuracy Scores: {cv_scores}")
# print(f"MLPClassifier 5-Fold CV Mean Accuracy: {cv_scores.mean():.4f}")

# # Eğitim seti ile eğit, test seti ile değerlendir
# mlp.fit(X_train, y_train)
# mlp_test_preds = mlp.predict(X_test)
# print("\nMLPClassifier Test Sonuçları:")
# print(classification_report(y_test, mlp_test_preds))
# print(f"MLP Test Accuracy: {accuracy_score(y_test, mlp_test_preds):.4f}")

# MLP için parametre arama
param_grid = {
    'hidden_layer_sizes': [(64,), (128, 64)],
    'activation': ['relu', 'tanh'],
    'max_iter': [300, 500],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.01],
    'batch_size': ['auto', 32, 64],
    'early_stopping': [True, False],

}
mlp = MLPClassifier(random_state=42)
grid_search = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f"En iyi parametreler: {grid_search.best_params_}")
print(f"En iyi eğitim skoru: {grid_search.best_score_:.4f}")

# En iyi model ile test seti değerlendirmesi
best_mlp = grid_search.best_estimator_
y_pred = best_mlp.predict(X_test)
print("\nTest Sonuçları (En iyi MLP):")
print(classification_report(y_test, y_pred))
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

