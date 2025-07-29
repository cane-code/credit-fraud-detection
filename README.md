# Retyping final evaluation procedures by 
    Mauludil Asri M Cane | 24.55.1603
# Software Enginering Lecturer by 
    Robert Marco, S.T., M.T., Ph.D & Hanif Al Fatta, S.Kom., M.Kom., Ph.D
  
# Credit Card Fraud Detection - Random Forest

Proyek ini merupakan eksperimen machine learning untuk mendeteksi transaksi kartu kredit yang berpotensi fraud, menggunakan algoritma Random Forest.

## 📁 Struktur
- `src/model_rf.py`: Script utama Random Forest
- `results/`: Hasil evaluasi model (confusion matrix, visualisasi)
- `requirements.txt`: Library yang digunakan

## 🔧 Cara Menjalankan
```bash
pip install -r requirements.txt
python src/model_rf.py
```

## 📊 Hasil Evaluasi

Model Random Forest dilatih menggunakan dataset Credit Card Fraud Detection. Evaluasi dilakukan pada 30% data uji dengan hasil sebagai berikut:

- **Accuracy**: 99.94%
- **Precision**: 86.87%
- **Recall**: 79.05%
- **F1-Score**: 82.69%

📁 Lihat folder `results/` untuk:
- Confusion matrix (PNG)
- Class Distirbution
- F1-Score antar kelas
- Top 10 Feture Importance
- GridSearchCv
- Imbalanced Dataset
- Presisi Recal
- ROC Curve
- Grafik Perbandingan Setiap Skenario`

### 🔍 Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)
