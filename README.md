# 📈 USD/TRY Kur Tahmini - PyTorch LSTM Modeli

Bu proje, Yahoo Finance üzerinden son 5 yıllık **Dolar/TL (USD/TRY)** günlük kapanış verilerini çekerek, PyTorch tabanlı bir **LSTM (Long Short-Term Memory)** yapay sinir ağı modeli ile geleceğe dönük fiyat tahmini yapar.

Zaman serisi analizi ve derin öğrenmeye giriş yapmak, veya PyTorch ile LSTM mimarisinin nasıl kurulduğunu öğrenmek isteyenler için adım adım ve açıklayıcı bir yapıda tasarlanmıştır.

## 🚀 Özellikler

*   **Otomatik Veri Çekme:** `yfinance` kütüphanesi ile Yahoo Finance üzerinden güncel verileri otomatik indirir.
*   **Veri Ön İşleme:** Scikit-learn `MinMaxScaler` ile veriler 0-1 aralığına normalize edilir ve 60 günlük kayan pencereler (sliding window) oluşturulur.
*   **GPU Desteği:** Sistemde NVIDIA ekran kartı (CUDA) varsa otomatik olarak GPU üzerinde, yoksa CPU üzerinde çalışarak maksimum performans sağlar.
*   **Detaylı Çıktı:** Ortalama Mutlak Hata (MAE) ve Kök Ortalama Kare Hata (RMSE) metriklerini hesaplar.
*   **Görselleştirme:** Matplotlib ile gerçek değerler ve modelin tahminlerini karşılaştıran bir grafik çizer.

## 🛠️ Kullanılan Teknolojiler

*   **Python 3.x**
*   **PyTorch** (Derin Öğrenme Modeli)
*   **yfinance** (Finansal Veri Seti)
*   **Pandas & NumPy** (Veri Manipülasyonu)
*   **Scikit-Learn** (Veri Ölçeklendirme ve Hata Metrikleri)
*   **Matplotlib** (Grafik Çizimi)

## 📦 Kurulum

Projeyi bilgisayarınıza klonladıktan veya indirdikten sonra, gerekli kütüphaneleri kurmak için terminalinizde şu adımları izleyin:

1. Gerekli paketleri içeren `requirements.txt` dosyasını oluşturun (veya indirin):
   ```text
   yfinance
   pandas
   numpy
   scikit-learn
   torch
   matplotlib
