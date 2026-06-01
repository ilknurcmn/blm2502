import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

print('Importing datasets...')

df_human = pd.read_csv('man_walking.csv', header=None)
df_sphere = pd.read_csv('unit_sphere.csv', header=None)
df_box = pd.read_csv('unit_box.csv', header=None)

df_box['label'] = 0
df_sphere['label'] = 1
df_human['label'] = 2

df_all = pd.concat([df_box, df_sphere, df_human], ignore_index=True)


print(f"Magnitude of aggregated datas : {df_all.shape} ")
print('Cleansing the dataset...')

df_all = df_all.replace(np.inf, 3.5)
df_all = df_all.fillna(3.5)

X = df_all.drop('label', axis=1)
y = df_all['label']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42, stratify=y)

print('Currently training the model, this may take a while...')

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Training is over. Accuracy : % {accuracy * 100:.2f}")

print("Detailed return:")

print(classification_report(y_test, y_pred, target_names=['Box' , 'Sphere', 'Human']))


#extra 

import matplotlib.pyplot as plt

print("\nModelin Karar Mekanizması Çözümleniyor...")
importances = model.feature_importances_

# En önemli 10 lazer açısını bulma
indices = np.argsort(importances)[::-1]
print("En yüksek bilgi taşıyan ilk 10 LiDAR açısı:")
for i in range(10):
    print(f"{i+1}. Açı (Sütun {indices[i]}): Etki Oranı % {importances[indices[i]] * 100:.2f}")

# Görselleştirme
plt.figure(figsize=(10, 5))
plt.plot(importances, color='red')
plt.title("360 Derece LiDAR Lazerlerinin Karar Verme Ağırlıkları")
plt.xlabel("Lazer Açısı (0-359)")
plt.ylabel("Önem Derecesi (Gini Importance)")
plt.grid(True)
plt.show()