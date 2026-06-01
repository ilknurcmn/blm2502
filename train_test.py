import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

csv_dir = "raw_csv_datas"

def extract_csv(file_path, label_name):
    try:
        df = pd.read_csv(file_path, header=None)
        infs_per_row = (df == np.inf).sum(axis=1)
        df_clean = df[infs_per_row<=350]

        features = []
        for index,row in df_clean.iterrows():
            valid_points = row[row != np.inf]
            if len(valid_points) == 0: 
                continue

            features.append({
                'Approx_of_laser' : len(valid_points),
                'Minimum_space' : valid_points.min(),
                'Avg_distance' : valid_points.mean(),
                'Surface_smoothness' : valid_points.std() if len(valid_points) > 1 else 0.0,
                'Label' : label_name
            })

        return pd.DataFrame(features)
    except Exception as e:
        return pd.DataFrame()
    

label_dict = {
    'suv.csv': 'SUV', 'hatchback_car.csv': 'Hatchback', 'barrier.csv': 'Barrier',
    'wheel.csv': 'Wheel', 'cardboard_box.csv': 'Cardbox', 'barrel.csv': 'Barrel',
    'dumpster.csv': 'Dumpster', 'traffic_cone.csv': 'Cone', 'man_walking.csv':'ManWalking',
    'cinder_block.csv': 'CinderBlock', 'postbox.csv': 'PostBox', 'sign.csv': 'Sign',
    'unit_box.csv':'Wall', 'unit_sphere.csv':'Sphere'
}

print("Exporting and seperating datas...--------------/")

all_datas = []
for filename in os.listdir(csv_dir):
    if filename.endswith(".csv") and filename in label_dict:
        df_extracted = extract_csv(os.path.join(csv_dir, filename), label_dict[filename])
        if not df_extracted.empty: all_datas.append(df_extracted)


master_df = pd.concat(all_datas, ignore_index=True)
attr_names = ['Approx_of_laser','Minimum_space','Avg_distance','Surface_smoothness']

X = master_df[attr_names]
y = master_df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"\nAccuracy: % {accuracy_score(y_test, y_pred) * 100:.2f}\n")
print(classification_report(y_test, y_pred))


# --- ÖZELLİK ÖNEMİ GRAFİĞİ ÇİZDİRME ---
importances = model.feature_importances_

plt.figure(figsize=(10, 6))
bars = plt.bar(attr_names, importances * 100, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title("What effects the models choices?", fontsize=14, fontweight='bold')
plt.ylabel("Decision change imporrtance(%)", fontsize=12)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"%{yval:.1f}", ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


import joblib

joblib.dump(model, 'random_forest_model.pkl')

print("Successfully saved to SSD")