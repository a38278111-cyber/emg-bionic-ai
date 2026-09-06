import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Datasetni yuklash (ajratuvchi belgi ';' va o'nlik belgi ',' ekanligini hisobga olamiz)
print("⏳ Ma'lumotlar yuklanmoqda...")
df = pd.read_csv('emg_dataset.csv', sep=';', decimal=',')

# 2. Xususiyatlar (X) va Maqsadli sinfni (y) ajratish
# O'ZGARISH SHU YERDA: Fayl_Nomi ustuni yo'qligi uchun faqat Target_Class ni o'chiramiz
X = df.drop(columns=['Target_Class'])
y = df['Target_Class']

# 3. Datasetni O'rgatish (80%) va Test (20%) qismlariga ajratish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Ma'lumotlarni Normalizatsiya qilish (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Random Forest modelini yaratish va o'rgatish
print("🧠 Model o'rgatilmoqda (Random Forest)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 6. Modelni test ma'lumotlarida sinab ko'rish va natijalarni hisoblash
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"✅ Model muvaffaqiyatli o'rgatildi!")
print(f"🎯 Umumiy aniqlik (Accuracy): {accuracy * 100:.2f}%")
print("="*50)
print("\nBatafsil Hisobot (Classification Report - amaliy ishga yozish uchun):\n")
print(classification_report(y_test, y_pred))

# 7. O'rgatilgan model va scaler'ni loyiha papkasiga saqlash
joblib.dump(rf_model, 'emg_rf_model.pkl')
joblib.dump(scaler, 'emg_scaler.pkl')
print("\n💾 'emg_rf_model.pkl' va 'emg_scaler.pkl' fayllari saqlandi!")