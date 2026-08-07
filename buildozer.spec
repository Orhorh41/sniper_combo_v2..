[app]
title = Kripto Sinyal Tarayici
package.name = kriptosinyal
package.domain = org.kullanici

source.dir = .
source.include_exts = py,png,jpg,kv,atlas

version = 1.0

# pandas KASITLI OLARAK YOK — Android derlemesinde en sık patlayan
# bağımlılıktır. Tüm hesaplamalar numpy ile yapılıyor (main.py içinde).
requirements = python3,kivy==2.2.1,requests,numpy,certifi,urllib3,idna,charset_normalizer,openssl
orientation = portrait
fullscreen = 0

icon.filename = %(source.dir)s/icon.png

android.permissions = INTERNET,ACCESS_NETWORK_STATE,WAKE_LOCK

# Android API/NDK seviyeleri — buildozer varsayılanlarıyla uyumlu, güncel
android.api = 34
android.minapi = 24
android.ndk = 25b
android.accept_sdk_license = True

# Arka planda tarama + pozisyon takibi thread'leri sürekli çalıştığı için
# uygulama arka plana alındığında Android'in işlemi öldürmesini engellemeye
# yardımcı olur (WAKE_LOCK izniyle birlikte).
android.wakelock = True

[buildozer]
log_level = 2
warn_on_root = 1
