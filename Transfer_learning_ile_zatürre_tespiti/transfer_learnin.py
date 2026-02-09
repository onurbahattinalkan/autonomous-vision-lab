"""
zatürre siniflandirma icin transfer öğrenme modeli
zatürre : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

transfer learning model : denset121

"""
# import libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator # goruntu verisi yukleme ve data augmentasyonu
from tensorflow.keras.applications import DenseNet121 # onceden egitilmis model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout # model katmanlari
from tensorflow.keras.models import Model # model olusturma
from tensorflow.keras.optimizers import Adam # optimizasyon algoritmasi
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # callbackler

import matplotlib.pyplot as plt # veri gorsellestirme
import numpy as np # sayisal islemler   
import os # dosya islemleri
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # karisiklik matrisi ve gorsellestirme


# load data and data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1/255.0, # normalizasyon 0-1 arasi
    horizontal_flip=True, # yatay cevirme
    rotation_range=10, # +- 10 derece donme
    brightness_range=[0.8,1.2], # parlaklik degisimi
    validation_split=0.1 # egitim ve dogrulama icin bolme
) # train data = train + validation

test_datagen = ImageDataGenerator(rescale=1/255.0) # test verisi icin sadece normalizasyon

DATA_DIR = "archive/chest_xray" # veri seti dizini
IMG_SIZE = (224,224) # modelin bekledigi girdi boyutu
BATCH_SIZE = 64 # batch size
CLASS_MODE = "binary" # binary siniflandirma

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR,"train"), # egitim verisinin bulundugu klasor
    target_size=IMG_SIZE, # goruntuleri image size'a yeniden boyutlandirma
    batch_size=BATCH_SIZE, # batch size
    class_mode=CLASS_MODE, # binary siniflandirma
    subset="training", # egitim alt kumesi
    shuffle=True # vreiyi karistirma
)

val_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR,"train"), # validation verisinin bulundugu klasor
    target_size=IMG_SIZE, # goruntuleri image size'a yeniden boyutlandirma
    batch_size=BATCH_SIZE, # batch size
    class_mode=CLASS_MODE, # binary siniflandirma
    subset="validation", # validation alt kumesi
    shuffle=False # validation verisi sirali olmalidir
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR,"test"), # test verisinin bulundugu klasor
    target_size=IMG_SIZE, # goruntuleri image size'a yeniden boyutlandirma
    batch_size=BATCH_SIZE, # batch size
    class_mode=CLASS_MODE, # binary siniflandirma
    shuffle=False # test verisi sirali olmalidir
)



# basic visualization

class_names = list(train_gen.class_indices.keys()) # sinif isimleri ['NORMAL', 'PNEUMONIA']
images, labels = next(train_gen) # bir batch veri al

plt.figure(figsize=(10,4))
for i in range(4):
    ax = plt.subplot(1,4,i+1)
    ax.imshow(images[i]) # goruntuyu goster
    ax.set_title(class_names[int(labels[i])]) # sinif ismini baslik olarak ekle
    ax.axis("off") # eksenleri kapat
plt.tight_layout()
plt.show()

# transfer learning modelin tanimlanmasi : DenseNet121
base_model = DenseNet121(
    weights = "imagenet", # onceden egitilmis agirliklar
    include_top = False, # son katmanlari dahil etme
    input_shape = (*IMG_SIZE, 3) # girdi boyutu (224,224,3)
)
base_model.trainable = False # base modeli dondur yani base model train edilmeyecek

x = base_model.output # base modelin cikistisi
x = GlobalAveragePooling2D()(x) # global average pooling katmani ekle
x = Dense(128, activation="relu")(x) # 128 nöronlu gizli katman
x = Dropout(0.5)(x) # overfittingi onlemek icin dropout katmani
pred = Dense(1, activation="sigmoid")(x) # cikis katmani binary siniflandirma icin sigmoid aktivasyonu

model = Model(inputs=base_model.input, outputs=pred) # modeli tanimla

# modelin derlenmesi ve callback ayarlari 

model.compile(
    optimizer=Adam(learning_rate= 1e-4), # Adam optimizasyon algoritmasi
    loss = "binary_crossentropy", # binary crossentropy kayip fonksiyonu
    metrics = ["accuracy"] # dogruluk metriği
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True), # erken durdurma
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6), # ogrenme oranini azaltma
    ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True), # en iyi modeli kaydetme
]
print("Model Summary:")
print(model.summary()) # model ozeti

# modelin egitilmesi ve sonuclarin degerlendirilmesi

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=2,
    callbacks=callbacks,
    verbose=1
)

pred_probs = model.predict(test_gen, verbose = 1) # test verisi uzerinde tahmin yapma
pred_labels = (pred_probs > 0.5).astype(int).ravel() # 0.5'ten buyuk olasiliklari 1, kucuk olasiliklari 0 olarak siniflandirma
true_labels = test_gen.classes # gercek etiket verilerimiz

cm = confusion_matrix(true_labels, pred_labels) # karisiklik matrisi
disp = ConfusionMatrixDisplay(cm, display_labels=class_names) # karisiklik matrisi gorsellestirme

plt.figure(figsize=(6,6))
disp.plot(cmap = "Blues", colorbar=False)
plt.title("Test Seti Confusion Matrix")
plt.tight_layout()
plt.show()