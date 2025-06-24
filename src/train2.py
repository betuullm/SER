import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
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
   
# Hangi verinin hangi labela denk geldiğini konsolda listele
for i in range(10):  
    print(f"{features.iloc[i]['file']} ({features.iloc[i]['folder']}) -> Label: {features.iloc[i]['label']}")
print(f"... Toplam {len(features)} kaydın ilk 10'u gösterildi ...")

for i in range(len(features)-10, len(features)):
    print(f"{features.iloc[i]['file']} ({features.iloc[i]['folder']}) -> Label: {features.iloc[i]['label']}")
print(f"... Toplam {len(features)} kaydın son 10'u gösterildi ...")

# Özellik sütunlarını seç (son 3 sütun: file, folder, label hariç)
X = features.iloc[:, :-3].values
y = features["label"].values.astype(int)

# Özellikleri ölçekle
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 85-15 oranında train ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Eğitim veri seti boyutu: {X_train.shape[0]}")
print(f"Test veri seti boyutu: {X_test.shape[0]}")
print(f"Toplam veri: {features.shape[0]} kayıt, {X.shape[1]} özellik")

mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', alpha=0.001, learning_rate='constant', max_iter=200, random_state=42)

# 5-fold cross validation (sadece eğitim verisi ile)
cv_scores = cross_val_score(mlp, X_train, y_train, cv=5)
print(f"\nMLPClassifier 5-Fold CV Accuracy Scores: {cv_scores}")
print(f"MLPClassifier 5-Fold CV Mean Accuracy: {cv_scores.mean():.4f}")

# Eğitim seti ile eğit, test seti ile değerlendir
mlp.fit(X_train, y_train)
mlp_test_preds = mlp.predict(X_test)
print("\nMLPClassifier Test Sonuçları:")
print(classification_report(y_test, mlp_test_preds))
print(f"MLP Test Accuracy: {accuracy_score(y_test, mlp_test_preds):.4f}")

