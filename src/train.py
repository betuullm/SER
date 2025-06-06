import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Her kayda sırayla Labels.csv'den etiket ata
features = pd.read_csv("..//data//train_features.csv")
labels = pd.read_csv("..//data//Labels.csv")
# Labels.csv'deki 172 etiketi, toplam kayıt sayısına ulaşana kadar döngüyle tekrar et
num_labels = len(labels)
repeats = len(features) // num_labels
remainder = len(features) % num_labels
full_labels = list(labels["Label"].values) * repeats + list(labels["Label"].values)[:remainder]
features["label"] = full_labels

# Hangi verinin hangi labela denk geldiğini konsolda listele
for i in range(10):  # İlk 10 örnek için göster
    print(f"{features.iloc[i]['file']} ({features.iloc[i]['folder']}) -> Label: {features.iloc[i]['label']}")
print(f"... Toplam {len(features)} kaydın ilk 10'u gösterildi ...")

# Özellik sütunlarını seç (son 3 sütun: file, folder, label hariç)
X = features.iloc[:, :-3]
y = features["label"]

# Train-test böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modeli eğit
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Tahmin ve değerlendirme
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
