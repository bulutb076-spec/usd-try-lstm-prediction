import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 5. ADIM: LSTM MODEL MİMARİSİ ---
class KurTahminLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(KurTahminLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Katmanı: batch_first=True diyerek (Batch, Sequence, Feature) yapısını tanıtıyoruz.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Çıktı (Linear) Katmanı
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Başlangıçtaki hafıza durumlarını sıfır matrisi ile başlatıyoruz.
        # x.device kullanarak modelin bulunduğu cihaza (CPU veya GPU) otomatik uyum sağlamasını garantiliyoruz.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Veriyi LSTM'den geçiriyoruz
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Sadece son günün (60. günün) çıktısını alıyoruz
        out = self.fc(out[:, -1, :]) 
        return out


def main():
    # --- Cihaz (Device) Kontrolü ---
    # Eğer bilgisayarda uygun bir ekran kartı (GPU) varsa onu kullanır, yoksa işlemciyle (CPU) devam eder.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Donanım (Device): {device}\n")

    print("--- 1. ve 2. ADIM: VERİ İNDİRME VE İNCELEME ---")
    df = yf.download('TRY=X', period='5y', interval='1d')
    df = df[['Close']].dropna() # Sadece kapanışı al ve olası eksik verileri (NaN) güvenli şekilde sil
    
    print("\nEksik Veri Kontrolü:")
    print(df.isnull().sum())
    print("\nİlk 5 Satır:")
    print(df.head())

    print("\n--- 3. ADIM: VERİ ÖN İŞLEME BAŞLIYOR ---")
    dataset = df.values 

    # 1. Normalizasyon
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # 2. Sliding Window (Kayan Pencere) Oluşturma
    window_size = 60
    X, y = [], []

    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 3. Train / Test Bölünmesi
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Eğitim Seti (X_train): {X_train.shape}")
    print(f"Test Seti (X_test): {X_test.shape}")

    print("\n--- 4. ADIM: PYTORCH DATALOADER HAZIRLIĞI ---")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\n--- 5. ADIM: LSTM MODEL KURULUMU ---")
    # Modeli oluşturup belirlediğimiz cihaza (GPU/CPU) gönderiyoruz
    model = KurTahminLSTM().to(device) 
    print(model)

    print("\n--- 6. ADIM: MODEL EĞİTİMİ (TRAINING) ---")
    learning_rate = 0.001
    num_epochs = 50

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Eğitim Başlıyor... Hatadaki (Loss) düşüşü izle:")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            # Batch verilerini de cihazımıza gönderiyoruz
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            tahmin = model(batch_X)
            loss = criterion(tahmin, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            ortalama_hata = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Ortalama Hata (Loss): {ortalama_hata:.6f}")

    print("Eğitim Tamamlandı!")

    print("\n--- 7. ADIM: DEĞERLENDİRME VE GÖRSELLEŞTİRME ---")
    model.eval()
    with torch.no_grad():
        # Test tensorünü modele sokarken cihaza alıyoruz. 
        # Çıkan sonucu numpy'a çevirmeden önce .cpu() ile sistem belleğine geri döndürüyoruz.
        test_tahminleri = model(X_test_tensor.to(device)).cpu().numpy()

    gercek_degerler = y_test_tensor.numpy()

    # Gerçek fiyatlara geri dönüş
    test_tahminleri_gercek = scaler.inverse_transform(test_tahminleri)
    gercek_degerler_gercek = scaler.inverse_transform(gercek_degerler)

    # Hata metrikleri
    mae = mean_absolute_error(gercek_degerler_gercek, test_tahminleri_gercek)
    rmse = np.sqrt(mean_squared_error(gercek_degerler_gercek, test_tahminleri_gercek))

    print(f"Ortalama Mutlak Hata (MAE): {mae:.4f} TL")
    print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.4f} TL")

    # Grafik Çizimi
    plt.figure(figsize=(12, 6))
    plt.plot(gercek_degerler_gercek, label='Gerçek USD/TRY Kuru', color='blue', linewidth=2)
    plt.plot(test_tahminleri_gercek, label='LSTM Tahmini', color='red', linestyle='dashed', linewidth=2)
    plt.title('USD/TRY Kapanış Fiyatı Tahmini (Test Seti Üzerinde)')
    plt.xlabel('Zaman (Test Setindeki Günler)')
    plt.ylabel('Fiyat (TL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Bu blok, kodun başka bir yerden import edilmesi durumunda otomatik çalışmasını engeller, 
# sadece dosya doğrudan çalıştırıldığında içindeki kodları tetikler (Profesyonel standart).
if __name__ == "__main__":
    main()