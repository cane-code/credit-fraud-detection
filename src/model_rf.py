import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# Load Dataset
# ========================
DATA_PATH = 'data/creditcard.csv'

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

df.head()

# Menampilkan data statistik Dataset
df.describe()

# Cek data imbalance
print(df['Class'].value_counts())

# =========================================
# Visualisasi Persentase distribusi kelas
# =========================================


# Pisahkan fitur dan target
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Akan diSplit dengan 70% Data Training dan 30% Data Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


# Hitung persentase distribusi kelas
class_percent = df['Class'].value_counts(normalize=True) * 100

# Buat DataFrame
percent_df = pd.DataFrame({
    'Class': ['Not Fraud (0)', 'Fraud (1)'],
    'Percentage': class_percent.sort_index().values
})

# membuat Visualisasi
plt.figure(figsize=(6, 4))
sns.barplot(x='Class', y='Percentage', hue='Class', data=percent_df, palette='Set2', legend=False)
plt.title('Class Distribution Percentage (Imbalanced Data)')
plt.ylabel('Percentage (%)')
plt.ylim(0, 110)

# Tambahkan nilai persentase di atas batang
for i, val in enumerate(percent_df['Percentage']):
    plt.text(i, val + 1, f"{val:.2f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ========================
# Preprocessing
# ========================

# Pisahkan fitur dan target
print("Preprocessing...")
X = df.drop(['Class'], axis=1)
y = df['Class']

# Data Akan diSplit dengan 70% Data Training dan 30% Data Testing
# ---------Data yang diterapkan SMOTE adalah pada data training saja untuk lebih membuat ampuh model, 
# ---------Karna saat di coba dengan data tes dia memeriksa dengan data buta, sehingga membuat model lebih valid 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Sebelum menerapkan SMOTE dataset dilakukan Standarscalar untuk fitur "Time" dan "Amount"
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

scaler = StandardScaler()

# Simpan indeks kolom Time dan Amount , dilakukan pada data training dengan fit
time_amount_cols = ['Time', 'Amount']
X_train[time_amount_cols] = scaler.fit_transform(X_train[time_amount_cols])
X_test[time_amount_cols] = scaler.transform(X_test[time_amount_cols])



# ========================
# Model Training
# ========================

# >>>>>>>>>>>>> Skenario 1 Dengan Baseline RandomForestClassifier <<<<<<<<<<<<<

# Latih model tanpa SMOTE dan tanpa tuning
rf_baseline = RandomForestClassifier(random_state=42, class_weight=None)
rf_baseline.fit(X_train, y_train)

# Prediksi
y_pred_1 = rf_baseline.predict(X_test)
y_proba_1 = rf_baseline.predict_proba(X_test)[:, 1]

print("Confusion Matrix Baseline:")
print(confusion_matrix(y_test, y_pred_1))

print("\nClassification Report  Baseline:")
print(classification_report(y_test, y_pred_1, digits=4))

print("\nROC AUC Score Baseline:")
print(roc_auc_score(y_test, y_proba_1))

# ============================================================
# Hasil Skenario 1
# ============================================================
#Confusion Matrix Baseline:
#[[85290     5]
# [   36   112]]

#Classification Report  Baseline:
#              precision    recall  f1-score   support

#           0     0.9996    0.9999    0.9998     85295
#           1     0.9573    0.7568    0.8453       148

#    accuracy                         0.9995     85443
#   macro avg     0.9784    0.8783    0.9225     85443
#weighted avg     0.9995    0.9995    0.9995     85443

#ROC AUC Score Baseline:
#0.9307385892839318

# =============================================================

# >>>>>>>>>>>>> Skenario 2 Penerapan SMOTE <<<<<<<<<<<<<
# Menerapkan SMOTE hanya pada training set
smote = SMOTE(random_state=42) # memberikan nilai K_n lebih rendah karna minoritas lebih sedikit

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Hitung distribusi sebelum dan sesudah SMOTE
import pandas as pd

original_dist = y_train.value_counts(normalize=True) * 100
smote_dist = pd.Series(y_train_smote).value_counts(normalize=True) * 100

#===========================================
#Visualisasi sebelum dan sesudah smote
#===========================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Dataframe dist_df harus sudah ada sebelumnya

plt.figure(figsize=(8, 5))
ax = sns.barplot(x='Class', y='Percentage', hue='Dataset', data=dist_df, palette='Set2')

plt.title('Class Distribution Before and After SMOTE')
plt.ylabel('Percentage (%)')
plt.ylim(0, 110)  # beri ruang di atas batang agar label tidak menabrak

# Tambahkan label angka di atas batang
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 5),  # geser ke atas 5 pt
                textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold')

plt.legend(title='Dataset')
plt.tight_layout()
plt.show()

# ===============================================================================
# Hasil dilihat pada Direktori:  Results/class_distribution_before_afte_SMOTE.png
# ===============================================================================

#===============================================================================
#Mulai melakukan Pelatihan Enhanced Random Forest Classifie
#Konfigurasi Enhanced Random Forest class_weight='balanced' → bantu model memperhatikan kelas minoritas.

#n_estimators=100 → jumlah pohon.
#random_state=42 → hasil konsisten.
#Gunakan data hasil SMOTE untuk pelatihan.
#================================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Inisialisasi dan latih Random Forest
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_smote, y_train_smote)

# ==============================
# Model Skenario 2 random forest
# ==============================
model_no_smote = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model_no_smote.fit(X_train, y_train)

#Prediksi pada test set
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# =================================
# Evaluasi Skenario 2

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nROC AUC Score:")
print(roc_auc_score(y_test, y_proba))
# ========================================

# =======================================================
# Hasil Evaluasi skenario 2
# =======================================================
#Confusion Matrix:
#[[85276    19]
# [   31   117]]

#Classification Report:
#              precision    recall  f1-score   support

#           0     0.9996    0.9998    0.9997     85295
#           1     0.8603    0.7905    0.8239       148

#    accuracy                         0.9994     85443
#   macro avg     0.9300    0.8952    0.9118     85443
#weighted avg     0.9994    0.9994    0.9994     85443

# ROC AUC Score:
# 0.9511345758678545

# ========================================================

# Visualisasi Confusion Matrik
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Buat confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Buat heatmap dari confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Fraud (0)', 'Fraud (1)'],
            yticklabels=['Not Fraud (0)', 'Fraud (1)'])

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# =================================================================================
# Hasil confusion matrik bisa dilihat pada direktori : results/confusion_matrix.png
# ==================================================================================

# =================================================================
# Visualissasi ROC Curve 
# =================================================================

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Asumsikan Anda sudah punya:
# y_test  → label asli dari data uji
# y_proba → probabilitas prediksi kelas 1 dari model terbaik
# Contoh: y_proba = model.predict_proba(X_test)[:, 1]

# Hitung ROC Curve dan AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

# Plot ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# =================================================================================
# Hasil visualisasi ROC Curve bisa dilihat pada direktori : results/roc_curva.png
# ==================================================================================

# =================================================================
# Visualissasi Precision-Recall Curve
# =================================================================
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Asumsikan Anda sudah punya:
# y_test  → label asli dari data uji
# y_proba → probabilitas prediksi kelas 1 dari model terbaik

# Hitung Precision-Recall Curve dan Area Under Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

# Plot Precision-Recall Curve
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================================================================================================
# Hasil visualisasi Precision-Recall Curve bisa dilihat pada direktori : results/presisi_recal.png.png
# ======================================================================================================


# ==================================================
# Model Skenario 3 Tuned RF (Tuned Random Forest)
# =====================================================
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Split data sudah dilakuan
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# pnerapan SMOTE sudah dilakukan pada skenario 2
#smote = SMOTE(random_state=42) 
#X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Lakukan Tuning dengan GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}

# Pelatihan Model
rf_turned = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_turned, param_grid, scoring='f1', cv=3)
grid_search.fit(X_train_smote, y_train_smote)

# Hasil pelatihan model dapar pada direktori : results/GridSearchCV.PNG


# Ambil dan Gunakan model terbaik dari Grid Search pada skeanrio 3
best_model = grid_search.best_estimator_

# Prediksi Skenario 3
y_pred_3 = best_model.predict(X_test)
y_proba_3 = best_model.predict_proba(X_test)[:, 1]

#======================
#Evaluasi
#======================
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_3))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_3, digits=4))

print("\nROC AUC Score:")
print(roc_auc_score(y_test, y_proba_3))

# ===============================
# Hasil Evaluasi 

# Confusion Matrix:
#[[85277    18]
# [   31   117]]

#Classification Report:
#              precision    recall  f1-score   support

#           0     0.9996    0.9998    0.9997     85295
#           1     0.8667    0.7905    0.8269       148

#    accuracy                         0.9994     85443
#   macro avg     0.9332    0.8952    0.9133     85443
#weighted avg     0.9994    0.9994    0.9994     85443

#ROC AUC Score:
#0.9617481776283582

# ====================================================

# ====================================================
# Visualisasi Bar Chart F1-Score antar Skenario
# ====================================================
import matplotlib.pyplot as plt

# Contoh data hasil F1-score tiap skenario (ganti dengan data asli Anda)
skenario_nama = ['Without SMOTE', 'With SMOTE', 'Tuned RF']
f1_scores = [0.8453,  0.8239, 0.8269]  # F1-score untuk kelas 1 (Fraud)

# Plot bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(skenario_nama, f1_scores, color=['#FF9999', '#66B3FF', '#99FF99'])

# Tambahkan label nilai di atas batang
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}',
             ha='center', va='bottom', fontweight='bold')

plt.title('Comparison of F1-Score of Fraud Classes Between Scenarios')
plt.ylabel('F1-Score')
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# ======================================================================================================
# Hasil visualisasi Bar Char bisa dilihat pada direktori : results/F1_score_antar_kelas.png
# ======================================================================================================

# ================================================================
# Visualisasi Top 10 Feature Importance dari Model Random Forest


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Asumsikan best_model adalah model terbaik Anda (misalnya dari GridSearch)
# dan X_train_sm adalah fitur training yang digunakan untuk melatih model

# Ambil nama fitur
feature_names = X_train.columns  # gunakan X_train asli, bukan yang sudah SMOTE

# Ambil nilai pentingnya
importances = best_model.feature_importances_

# Buat DataFrame untuk sorting
feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Ambil 10 fitur teratas
top_feats = feat_df.head(10)

# Plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=top_feats, palette='viridis')
plt.title('Top 10 Feature Importance dari Model Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
# =========================================================================================================
# Hasil visualisasiTop 10 Feature Importance bisa dilihat pada direktori : results/feture_importance.png
# =========================================================================================================



