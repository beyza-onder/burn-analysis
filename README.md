# Yanık Analizi ve Sınıflandırma Sistemi 🚑

Bu proje, cilt görüntülerini analiz ederek yanık varlığını tespit eden ve yanıkları derecelendiren (1., 2. ve 3. Derece) bir Derin Öğrenme uygulamasıdır. Model, MobileNetV2 mimarisi kullanılarak geliştirilmiş ve Gradio kütüphanesi ile kullanıcı dostu bir arayüz sunulmuştur.
⚠️ Uyarı: Bu sistem bir yapay zeka tahminidir, kesin tanı amacıyla kullanılmamalıdır.

## 🌟 Özellikler
- **Doğrulama:** Sağlıklı cilt ile yanık dokusunu birbirinden ayırır.
- **Derecelendirme:** Yanıkları 1., 2. ve 3. derece olarak sınıflandırır.
- **İlk Müdahale Önerileri:** Tespit edilen yanık derecesine göre yapılması gereken ilk yardım adımlarını gösterir.
- **Güven Eşiği:** Modelin emin olmadığı durumlarda kullanıcıyı tıbbi yardım alması için uyarır.

## **Desteklenen görsel formatları:
*.jpg
*.jpeg
*.png
*.webp
*.bmp

## 🖥️Uygulama Arayüzü
- **Görüntü Yükle:** Kullanıcıya bir görsel yüklemesini sağlar.
- **Yanık Türü Tahmini:** Görsel analiz edilerek yanık türü tahmin edilir.
- **Güven Oranı (%):** Modelin tahmin güvenini gösterir.
- **İlk Müdahale Önerileri:** Tespit edilen yanık derecesine göre yapılması gereken ilk yardım adımlarını gösterir.
- **Belirsiz Sonuç Uyarısı:(confidence < %60)** Modelin emin olmadığı durumlarda kullanıcıyı tıbbi yardım alması için uyarır.


## 🛠️ Kullanılan Teknolojiler
- **Python**
- **TensorFlow/Keras** (MobilNetV2 tabanlı model)
- **Gradio** (Web arayüzü)
- **OpenCV / PIL** (Görüntü işleme)
- **Numpy & Scikit-learn** (Veri yönetimi ve sınıf ağırlıklandırma)

## **🏗️ Model Mimarisi
-MobileNetV2 (ImageNet ağırlıkları ile)
-Global Average Pooling
-Dense (256 nöron, ReLU)
-Dropout (%50)
-Softmax çıkış katmanı (4 sınıf)

Model, sınıf dengesizliğini azaltmak için class_weight kullanılarak eğitilmiştir

## **🧪 Model Eğitimi 
- **Görüntü boyutu:** 224 × 224
- **Epoch sayısı:** 35
- **Batch size:** 4
- **Optimizer:** Adam
- **Kayıp fonksiyonu:** Categorical Crossentropy

## 📁 Dosya Yapısı
Projeyi çalıştırmadan önce aşağıdaki klasör yapısının mevcut olduğundan emin olun:
```text
.
├── saglikli_cilt/          # Normal cilt görselleri
├── yanik_1derece/          # 1. derece yanık görselleri
├── yanik_2derece/          # 2. derece yanık görselleri
├── yanik_3decerece/        # 3. derece yanık görselleri
├── check_data.py           # Veri kontrolü dosyası
├── burn_classifier.py      # Ana uygulama dosyası
├── README.md               # Proje açıklaması
└── requirements.txt        # Gerekli kütüphaneler
```

## 🚀 Kurulum ve Çalıştırma

1. **Gerekli Kütüphaneleri Yükleyin:**
   ```bash
   pip install tensorflow pillow numpy gradio scikit-learn
   ```

2. **Uygulamayı Başlatın:**
   ```bash
   python burn_classifier.py
   ```

3. **Arayüze Erişin:**
   Terminalde çıkan `http://127.0.0.1:7860` adresini tarayıcınızda açarak sistemi kullanmaya başlayabilirsiniz.

## ⚠️ Önemli Uyarı
Bu uygulama yalnızca **eğitim ve bilgilendirme amaçlıdır.** Tıbbi bir teşhis aracı değildir. Ciddi yanıklarda veya emin olmadığınız durumlarda lütfen her zaman profesyonel bir sağlık kuruluşuna veya **112 Acil Çağrı Merkezi**'ne başvurun.

## **👩‍💻Geliştirici**
-**Beyza ÖNDER** <br>
-**Bilişim Sistemleri ve Teknolojileri Öğrencisi** <br>
-**Dijital Görüntü İşleme Projesi**
---

