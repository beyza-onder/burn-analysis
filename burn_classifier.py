import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr
from sklearn.utils import class_weight

# ---------------------------------------------------------
# Yanık Analizi Modeli (v5.1)
# ---------------------------------------------------------

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = {0: "saglikli_cilt", 1: "yanik_1derece", 2: "yanik_2derece", 3: "yanik_3decerece"}
    
    images, labels = [], []
    print("Veriler taranıyor...")
    for label, folder in dirs.items():
        folder_path = os.path.join(base_dir, folder)
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
            files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        print(f"{folder}: {len(files)} görsel bulundu.")
        for f in files:
            try:
                img = Image.open(f).convert('RGB').resize((224, 224))
                images.append(np.array(img).astype(float) / 255.0)
                labels.append(label)
            except: pass

    if not images: return None, None, None
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(weights))
    X = np.array(images)
    y = tf.keras.utils.to_categorical(np.array(labels), num_classes=4)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices], class_weights

X_train, y_train, class_weights = load_data()

# Model Yapılandırması
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Veri Artırma
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.3, horizontal_flip=True, fill_mode='nearest'
)

if X_train is not None:
    print("Model eğitiliyor...")
    model.fit(datagen.flow(X_train, y_train, batch_size=4), epochs=35, class_weight=class_weights, verbose=1)

def classify(image):
    if image is None: return "Lütfen görsel yükleyin."
    img = image.resize((224, 224))
    img_arr = np.expand_dims(np.array(img)/255.0, axis=0)
    preds = model.predict(img_arr)[0]
    idx = np.argmax(preds)
    conf = preds[idx]
    
    if conf < 0.60:
        return "⚠️ Analiz Sonucu Belirsiz: Görüntü net değil veya verilerle tam eşleşmiyor. Lütfen bir uzmana danışın."

    results = {
        0: ("SAĞLIKLI CİLT", 
            "Cilt normal görünüyor. Herhangi bir yanık belirtisi saptanmadı.",
            "Herhangi bir müdahaleye gerek yoktur. Cildinizi güneşten korumaya devam edin."),
        1: ("1. DERECE YANIK", 
            "Kızarıklık bazlı yüzeysel yanık (Güneş yanığı vb.).",
            "1. Bölgeyi hemen 10-20 dakika boyunca akan serin su altında tutun.\n2. Buz kullanmayın (dokuyu zedeleyebilir).\n3. Nemlendirici kremler veya Aloe Vera jeli kullanabilirsiniz."),
        2: ("2. DERECE YANIK", 
            "Su toplaması ve doku hasarı belirtileri mevcut.",
            "1. Bölgeyi 20 dakika boyunca serin su altında tutun.\n2. Su Keseciklerini (Bülleri) ASLA PATLATMAYIN.\n3. Temiz bir bezle örtün ve enfeksiyon riski için DOKTORA BAŞVURUN."),
        3: ("3. DERECE YANIK", 
            "Ağır doku hasarı, beyazlık veya kömürleşme mevcut.",
            "1. ACİL TIBBİ YARDIM ALIN (112).\n2. Yanığa kesinlikle hiçbir şey sürmeyin.\n3. Bölgeyi temiz bir bezle örtüp en yakın hastaneye gidin.")
    }
    
    name, desc, help_msg = results[idx]
    return (f"🔍 ANALİZ SONUCU: {name}\n"
            f"📊 Güven Oranı: %{conf*100:.1f}\n\n"
            f"📌 Durum: {desc}\n\n"
            f"🚒 İLK MÜDAHALE ÖNERİLERİ:\n{help_msg}\n\n"
            f"⚠️ Not: Bu bir yapay zeka tahminidir. Kesin tanı için sağlık kuruluşuna başvurun.")

# Arayüz
iface = gr.Interface(
    fn=classify, 
    inputs=gr.Image(type="pil", label="Görüntü Yükle"), 
    outputs=gr.Textbox(label="Analiz Sonucu ve İlk Müdahale", lines=10), 
    title="Yanık Analizi"
)

if __name__ == "__main__":
    iface.launch()
