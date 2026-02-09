""""
GAN(Generative Adversarial Network) fashion mnist veri seti ile moda urunu tasarimi yapalim

Fashion MNIST veri seti 10 sinif iceren 28x28 gri tonlamali goruntulerdir.
Fashion veri seti icresinide 10 farkli sinif bulunmaktadir.
- T-shirt/top, sneaker, bag, ankle boot, pullover, dress, coat, sandal, shirt, trouser


Plan Program?
Pip libraries
import libraries

"""
# import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import fashion_mnist


# veri seti yukle
(train_images, _),(_, _) = fashion_mnist.load_data() # sadece goruntuler gerekli etiketler gerekli degil
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') # sekillendir ve floata cevir
train_images = (train_images-127.5) / 127.5 # -1 ile 1 arasina normallestir
BUFFER_SIZE = 60000 # toplam goruntu sayisi
BATCH_SIZE = 128 # batch size
NOISE_DIM = 100 # generatora verilecek gürültü vektörünün boyutu
IMG_SHAPE = (28, 28, 1) # goruntu sekli
EPOCHS = 2

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE) # veri setini karistir ve batchlere ayir


# generator modeli tanimla : fake goruntu ureten model
def make_generator_model():
    model = keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_DIM,)), # ilk tam bagli katman, gurultuyu ozellik haritasina cevirir
        layers.BatchNormalization(), # egitim stabilitesi icin batch normalization
        layers.LeakyReLU(), # negatif girisleri yumusatir

        layers.Reshape((7, 7, 256)), # tek boyutlu vektrou 3D donustur ozellik haritasini 7x7x256 sekline getirir

        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False), # 128 filtreli transpoze konvolusyon katmani
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False), # 128 filtreli transpoze konvolusyon katmani
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh') # cikis katmani, 28x28x1 boyutunda goruntu uretir

    ])

    return model

# generator = make_generator_model()

def make_dicriminator_model():
    model = keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding= "same", input_shape= IMG_SHAPE), # 28x28x1 giris goruntusu icin konvolusyon katmani
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5,5), strides=(2,2), padding="same"), # ikinci konvolusyon katmani
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(), # tek boyutlu vektore donustur
        layers.Dense(1) # cikis katmani, gercek mi sahte mi karar verir


    ])

    return model

discriminator = make_dicriminator_model()
# discriminator modeli tanimla : gercek mi sahte mi ayiran model
# kayip fonksiyonlari ve optimizers tanimla 
cross_entropy = keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output) # gercek = 1 etiketine sahip olsun
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output) # sahte = 0 etiketine sahip olsun
    return real_loss + fake_loss # toplam discriminator kaybi

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output) # generator sahte goruntuler urettigi icin 1 etiketine sahip olsun isteriz

generator = make_generator_model()
discriminator = make_dicriminator_model()

generator_optimizer = keras.optimizers.Adam(1e-4) # generator optimizer
discriminator_optimizer = keras.optimizers.Adam(1e-4) # discriminator optimizer

# yardimci fonksiyonlari tanimla

seed = tf.random.normal([16, NOISE_DIM]) # sabit gurultu ornegi, egitim boyunca uretilen goruntuleri karsilastirmak icin kullanilir

def generate_and_save_images(model, epoch, test_input):
    prediction = model(test_input, training=False) # modeli sadece degerlendirme modunda kullanarak goruntu uret
    fig = plt.figure(figsize=(4,4)) # 4x4 lük bir figür olustur
    for i in range(prediction.shape[0]):
        plt.subplot(4, 4, i+1) # 4x4 lük subplot olustur
        plt.imshow((prediction[i, :, :, 0] + 1) / 2, cmap='gray') # goruntuyu -1 ile 1 arasindan tekrar 0 ile 1 arasina rescale et ve gri tonlamali olarak goster
        plt.axis('off') # eksenleri kapat

    if not os.path.exists("generated_images"):
        os.makedirs("generated_images") # generated_images adinda bir klasor olustur

    plt.savefig(f"generated_images/image_at_epoch_{epoch:03d}.png") # her epoch sonunda uretilen goruntuyu kaydet
    plt.close() # figuru kapat

# egitim fonksiyonlarini tanimla : gerçek ve sahte goruntulerle egit

def train(dataset,epoch):

    for epoch in range(1, epoch + 1):
        gen_loss_total = 0, # generator kayiplarini toplamak icin
        disc_loss_total = 0 # discriminator kayiplarini toplamak icin
        batch_count = 0 # batch sayisini takip etmek icin

        for image_batch in dataset:
            noise = tf.random.normal([BATCH_SIZE, NOISE_DIM]) # her batch icin yeni bir gurultu vektoru olustur
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape: # gradient hesaplamak icin iki tane tape olustur
                generated_images = generator(noise, training=True) # generatoru kullanarak sahte goruntuler uret

                real_output = discriminator(image_batch, training=True) # discriminatoru kullanarak gercek goruntuleri degerlendir
                fake_output = discriminator(generated_images, training=True) # discriminatoru kullanarak sahte goruntuleri degerlendir

                gen_loss = generator_loss(fake_output) # generator kaybini hesapla
                disc_loss = discriminator_loss(real_output, fake_output) # discriminator kaybini hesapla

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables) # generatorun egitilebilir degiskenlerine gore kaybin gradyanini hesapla
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables) # discriminatorun egitilebilir degiskenlerine gore kaybin gradyanini hesapla

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables)) # generatorun gradyanlarini optimizer ile uygula
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables)) # discriminatorun gradyanlarini optimizer ile uygula

            gen_loss_total += gen_loss # her batchin generator kaybini topla
            disc_loss_total += disc_loss # her batchin discriminator kaybini topla
            batch_count += 1 # batch sayisini arttir

        print(f"Epoch {epoch}/{epoch} - Generator Loss: {gen_loss_total / batch_count:.3f}, Discriminator Loss: {disc_loss_total / batch_count:.3f}") # her epoch sonunda ortalama kayiplari yazdir
        generate_and_save_images(generator, epoch, seed) # her epoch sonunda uretilen goruntuleri kaydet

train(train_dataset, EPOCHS) # modeli 50 epoch boyunca egit





