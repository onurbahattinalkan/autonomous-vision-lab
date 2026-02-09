"""
MNIST veri seti :
    rakamlama : 0-9 toplamda 10 sinif var
    28x28 piksel boyutunda resimler
    grayscale (siyah beyaz) görüntüler
    60000 eğitim örneği ve 10000 test örneği içerir
    amacimiz: ann ile bu resimlerdeki rakamlari tanimak ya da siniflandirmak

    Image processing: 
        histogram eşitleme :kontrast iyileştirme
        gaussian blur : gürültü azaltma
        canny edge detection : kenar tespiti

    ANN (artificial neural network)  ile MNIST veri setini siniflandirma

    libraries:
        tensorflow/keras : ann modeli oluşturma ve eğitme
        matplotlib : veri görselleştirme
        cv2: görüntü işleme
"""
# import libraries
import cv2
import numpy as np # sayisal islemler icin
import matplotlib.pyplot as plt # gorsellestirme icin

from tensorflow.keras.datasets import mnist #MNIST veri seti
from tensorflow.keras.models import Sequential # ann modeli icin
from tensorflow.keras.layers import Dense, Dropout # ann katmanlari icin
from tensorflow.keras.optimizers import Adam # optimizasyon icin

# load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
"""
x_train shape: (60000, 28, 28)
y_train shape: (60000,)
"""
"""
# image prepocessing
img = x_train[5] # ilk resmi al

stages={"orijinal": img}

# histogram eşitleme
eq = cv2.equalizeHist(img) # histogram esitleme
stages ["histogram esitleme"] = eq

# gaussian blur
blur = cv2.GaussianBlur(eq, (5,5),0) # gaussian blur
stages ["gaussian blur"] = blur

# canny ile kenar tespiti
edges = cv2.Canny(blur, 50,150) # kenar tespiti (50'nin altı kesin kenar degil, 150'nin ustu kesin kenar, aralık kesin kenar olabilir)
stages ["canny edge detection"] = edges

# gorsellestirme
fig, axes = plt.subplots(2,2, figsize=(6,6))
axes = axes.flat
for ax, (title, im) in zip(axes, stages.items()):
    ax.imshow(im, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.suptitle("MNIST Image Processing Stages")
plt.tight_layout()
plt.show()
"""
# preprocessing fonksiyonu

def preprocess_image(img):
    """
     -histogram esitleme
     -gaussian blur
     -canny ile kenar tespiti
     -flattering :28x28 -> 784 boyutuna cevirme
     -normalizasyon : 0-255 -> 0-1 arasi degerler

    """
    img_eq = cv2.equalizeHist(img)
    img_blur = cv2.GaussianBlur(img_eq, (5,5),0)
    img_edges = cv2.Canny(img_blur, 50,150)
    features = img_edges.flatten()/255.0 # 28x28 -> 784 ve normalizasyon
    return features
"""
num_train = 10000
num_test = 2000

x_train = np.array([preprocess_image(img) for img in x_train[:num_train]])
y_train_sub = y_train[:num_train]

x_test = np.array([preprocess_image(img) for img in x_test[:num_test]])
y_test_sub = y_test[:num_test]
"""

x_train = np.array([preprocess_image(img) for img in x_train])
y_train_sub = y_train[:]

x_test = np.array([preprocess_image(img) for img in x_test])
y_test_sub = y_test[:]

# ann model creation
model = Sequential ([
    Dense(128,activation='relu', input_shape=(784,)), #ilk katman, 128 nöron 28x28 = 784 boyutunda
    Dropout(0.5), # dropout katmani, overfitting'i onlemek icin
    Dense(64, activation='relu'), # ikinci katman, 64 nöron
    Dense(10, activation ='softmax') # cikis katmani, 10 sinif (0-9 rakamlari)
])

# compile model
model.compile(
    optimizer=Adam(learning_rate=0.001), #optimizer
    loss='sparse_categorical_crossentropy',# loss function
    metrics=['accuracy'] # performans metrikleri
)

print(model.summary())

# ann model training
history = model.fit(
    x_train, y_train_sub,
    validation_data=(x_test, y_test_sub),
    epochs=50,
    batch_size=32,
    verbose=2
)
# eveluate model performance
test_loss, test_acc = model.evaluate(x_test, y_test_sub)
print(f"test loss : {test_loss:.4f}, test accuracy: {test_acc:.4f}")

# plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


