"""
florwers dataset : 
    rgb: 224x224

CNN ile siniflandirma modeli olusturma ve problemi cozme

"""

# import libraries
from tensorflow_datasets import load # veri seti yukleme
from tensorflow.data import AUTOTUNE # veri seti optimizasyonu
from tensorflow.keras.models import Sequential # model olusturma
from tensorflow.keras.layers import (
    Conv2D, # 2D convolutional layer
    MaxPooling2D, # max pooling layer
    Flatten, # cok boyutlu veriyi tek boyutlu hale getirme
    Dense , # tam baglantili katman
    Dropout # rastgele noronları kapatarak overfittingi onleme
)
from tensorflow.keras.optimizers import Adam # optimizasyon algoritmasi
from tensorflow.keras.callbacks import (
    EarlyStopping, # erken durdurma
    ReduceLROnPlateau, # ogrenme oranini azaltma
    ModelCheckpoint # en iyi modeli kaydetme
)

import tensorflow as tf
import matplotlib.pyplot as plt # veri gorsellestirme


# veri seti yukleme

(ds_train, ds_val),ds_info = load(
    "tf_flowers",
    split=["train[:80%]", # veri setinin %80'i egitim icin
           "train[80%:]"], # veri setinin %20'si test icin
    as_supervised=True, # veri setinin gorsel ve etiket ciftinin olmasi
    with_info=True # veri seti hakkinda bilgi alma
)
print(ds_info.features)
print("Number of classes:", ds_info.features["label"].num_classes)

# ornek veri gorsellestirme
# egitim setinden rastgele 3 resim ve etiket alma
fig = plt.figure(figsize=(10,5))
for i, (image,label) in enumerate(ds_train.take(3)):
    ax = fig.add_subplot(1 ,3 ,i+1) # 1 satir , 3 sutun, i+1. resim
    ax.imshow(image.numpy().astype("uint8")) # resmi gorsellestirme
    ax.set_title(f"Etiket: {label.numpy()}") # etiket baslik olarak yazdirma
    ax.axis("off") # eksenleri kapatma

plt.tight_layout()
plt.show() # grafigi gosterme

IMG_SIZE = (180,180)

# data augmentation + preprocessing
def preprocess_train(image,label):
    """
    resize, randomfip, brightness, contrast, crop
    normalize
    """
    image = tf.image.resize(image,IMG_SIZE) # resmi yeniden boyutlandirma
    image = tf.image.random_flip_left_right(image) # yatay olarak rastgele cevirme
    image = tf.image.random_brightness(image, max_delta=0.1) # rastgele parlaklik
    image = tf.image.random_contrast(image, lower=0.9, upper=1.2) # rastgele kontrast
    image = tf.image.random_crop(image, size=(160,160,3)) # rastgele crop
    image = tf.image.resize(image, IMG_SIZE) # tekrar boyutlandirma
    image = tf.cast(image, tf.float32) / 255.0 # normalizasyon
    return image, label

def preprocess_val(image,label):
    """
    resize + normalize
    """
    image = tf.image.resize(image, IMG_SIZE) # boyutlandirma
    image = tf.cast(image, tf.float32) / 255.0 # normalizasyon
    return image, label

# veri setini hazirlamak
ds_train=(
    ds_train
    .map(preprocess_train, num_parallel_calls=AUTOTUNE) # on isleme ve augmentasyon
    .shuffle(1000) # karistirma
    .batch(32) # batch boyutu
    .prefetch(AUTOTUNE) # performans artisi
)

ds_val=(
    ds_val
    .map(preprocess_val, num_parallel_calls=AUTOTUNE) # on isleme
    .batch(32) # batch boyutu
    .prefetch(AUTOTUNE) # veri setini onceceden hazirlamak
)

# CNN modelini olsuturma
model = Sequential([

    # Feature Extraction
    Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE,3)), # 32 filtreli, 3x3 kernel detaylar ise küçük, relu aktivasyon, 3 kanal RGB
    MaxPooling2D((2,2)), # 2x2 max pooling

    Conv2D(64, (3,3), activation='relu'), # 64 filtreli, 3x3 kernel, relu aktivasyon, 3 kanal RGB
    MaxPooling2D((2,2)), # 2x2 max pooling

    Conv2D(128, (3,3), activation='relu'), # 128 filtreli, 3x3 kernel, relu aktivasyon, 3 kanal RGB
    MaxPooling2D((2,2)), # 2x2 max pooling

    # Classification
    Flatten(), # cok boyutlu veriyi tek boyutlu hale getirme
    Dense(128, activation='relu'), # 128 noronlu tam baglantili katman, relu aktivasyon
    Dropout(0.5), # %50 noron kapatma
    Dense(ds_info.features["label"].num_classes, activation='softmax') # cikis katmani, sinif sayisi kadar noron, softmax aktivasyon
])

# callbacks
callbacks = [
    # eger val_loss 3 epoch boyunca iyilesmezse egitimi durdur ve en iyi agirliklari yukle
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights = True,
    ),

    # val_loss 2 epoch boyunca iyilesmezse learning rate 0.2 carpani ile azalt
    ReduceLROnPlateau(
        monitor='val_loss', # izlenecek deger
        factor = 0.2, # ogrenme oranini azaltma carpani
        patience = 2, # sabir sayisi
        verbose = 1, # bilgi mesajlarini goster
        min_lr = 1e-9 # minimum ogrenme orani
    ),

    # ModelCheckpoint, # model kaydetme
    ModelCheckpoint(
        "best_model.h5", # kaydedilecek dosya adi
        save_best_only=True # en iyi modeli kaydet
    )
]

# derleme
model.compile(
    optimizer=Adam(learning_rate=0.001), # Adam optimizer, ogrenme oranini 0.001 olarak ayarla
    loss="sparse_categorical_crossentropy", # kayip fonksiyonu, etiketler tamsayi oldugu icin sparse_categorical_crossentropy kullanildi
    metrics=["accuracy"] # basari metriği olarak dogruluk kullan
)

print(model.summary()) # model ozeti

# training
history = model.fit(
    ds_train, # egitim veri seti
    validation_data=ds_val, # dogrulama veri seti
    epochs=10, # epoch sayisi
    callbacks=callbacks, # callbacks
    verbose=1 # egitim surecini goster
)

# model evaluation
plt.figure(figsize=(12,5))

# dogruluk grafigi
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Egitim Dogrulugu")
plt.plot(history.history["val_accuracy"], label="Validasyon Dogrulugu")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuarcy")
plt.legend()

#loss grafik
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Egitim Kaybi")
plt.plot(history.history["val_loss"], label="Validasyon Kaybi")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()

plt.tight_layout()
plt.show()








