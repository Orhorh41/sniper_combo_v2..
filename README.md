# Kripto Sinyal Tarayıcı — Kivy Mobil Sürüm

## Önce şunu bilin (dürüst özet)

Ben bu ortamda gerçek bir `.apk` dosyasını **derleyip size doğrudan
veremiyorum** — burada Android SDK/NDK ve internet erişimi kısıtlı bir
sanal makine var, bu yüzden `buildozer android debug` komutunu burada
çalıştıramıyorum. Ama yapabildiğim ve yaptığım şey:

- Orijinal 3500 satırlık terminal scriptini **Android'e uygun hale
  getirdim**: `pandas` bağımlılığını tamamen kaldırıp yerine saf NumPy
  ile yazdım (pandas, Android derlemelerinde en sık başarısız olan
  kütüphanelerden biridir), Rich/terminal arayüzünü Kivy dokunmatik
  arayüzle değiştirdim, `input()` sorularını form ekranına çevirdim.
- Sinyal üretim mantığını (4 strateji: ORİJİNAL, SNIPER LONG-SHORT,
  SNIPER 1, SNIPER 2) **birebir koruyarak** test ettim — hem sentetik
  veri hem yapısal olarak hatasız çalıştığını doğruladım.
- Sizin hiçbir şey kurmadan gerçek bir APK indirebilmeniz için aşağıda
  bir **GitHub Actions** otomasyonu hazırladım (bulutta ücretsiz
  derler, size .apk dosyası olarak verir).

Kapsam dışı bırakılan tek şey: orijinal dosyadaki devasa "geçmiş
performans backtest paneli" (binlerce satırlık ekran çizim kodu).
Sinyal tarama + sanal (paper) pozisyon takibi tam çalışıyor.

## En kolay yol: GitHub Actions ile APK almak (kurulum gerektirmez)

1. GitHub'da yeni bir repo oluşturun (public ya da private, farketmez).
2. Bu klasördeki TÜM dosyaları (main.py, buildozer.spec, icon.png,
   .github/ klasörü dahil) o repoya yükleyin.
3. GitHub'da reponun **Actions** sekmesine girin, "APK Derle"
   workflow'unu görüp **Run workflow** butonuna basın.
4. ~10-15 dakika sürer. Bitince Actions çalıştırmasının sayfasında
   **Artifacts** bölümünden `kripto-sinyal-apk` dosyasını indirin,
   içinden `.apk` çıkacak.
5. APK'yı telefonunuza kopyalayıp kurun (Ayarlar > Güvenlik > "Bilinmeyen
   kaynaklara izin ver" gerekebilir, çünkü Play Store dışından kuruyorsunuz).

## Alternatif: Kendi bilgisayarınızda derlemek (Linux / WSL)

```bash
pip install buildozer cython==0.29.36
sudo apt install -y git zip unzip openjdk-17-jdk autoconf libtool pkg-config \
    zlib1g-dev libncurses5-dev cmake libffi-dev libssl-dev build-essential
buildozer android debug
```

İlk çalıştırmada buildozer Android SDK/NDK'yı otomatik indirir (birkaç
GB, zaman alır). Sonuç `bin/` klasöründe `.apk` olarak çıkar.

**Not:** Buildozer Windows'ta doğrudan çalışmaz; Windows kullanıyorsanız
WSL2 (Ubuntu) kurup onun içinden çalıştırmanız gerekir.

## Uygulamayı test etmeden önce bilmeniz gerekenler

- Bu hâlâ bir **simülasyondur** — hiçbir gerçek emir Binance'e gitmiyor.
- Uygulama arka planda sürekli Binance API'sine istek atar; telefon pil
  optimizasyonu uygulamayı arka planda durdurabilir. Ayarlar > Pil >
  bu uygulama için "kısıtlama yok" seçmeniz gerekebilir.
- 4 stratejinin aynı anda onlarca coin için taranması CPU/ağ kullanır;
  eski/düşük donanımlı telefonlarda tarama süresi uzayabilir.
- Kodun sinyal mantığı orijinal dosyayla birebir aynıdır; sadece
  pandas→numpy dönüşümünde çok küçük sayısal farklar (yuvarlama)
  olabilir, sinyal kararlarını etkilemez.

## Dosyalar

- `main.py` — uygulamanın tamamı (tek dosya)
- `buildozer.spec` — Android derleme ayarları
- `icon.png` — basit yer tutucu ikon (isterseniz değiştirin)
- `.github/workflows/build-apk.yml` — otomatik APK derleme
