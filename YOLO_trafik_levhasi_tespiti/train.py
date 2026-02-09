"""
eo sensor : kamera, trafik kurallari, trafik isareterinin taninmasi
otonom aracin en temel gorevi cevreyi tanimak : isaretler (trafik levhalari)


plan program : veri bulma, veri yukleme , training , testing

data : https://universe.roboflow.com/university-km5u7/traffic-sign-detection-yolov8-awuus/dataset/11/download/yolov8
"""

from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

def main():
    # modeli sec yolov8n nano modeli
     model = YOLO("yolov8n.pt")  # once en light model ile baslayalim, daha sonra ihtiyac duyarsak daha buyuk modelleri kullanabiliriz
     
     model.train(
      data = "traffic-sign-detection/data.yaml", # yaml dosyasinda veri yollari ve sinif isimleri tanimli
      epochs = 10, # egitim dongü sayisi
      imgsz = 640, # girdi goruntu boyutu
      batch = 16, # mini batch boyutu donanima bağli ayarlanir
      name = "traffic-sign-model", # egitim sonucunda olusan modelin kaydedilecegi klasor adi, normalde buraya kadar yazilir gerisi default
      lr0 = 0.01, # baslangictaki ogrenme hizi
      optimizer = "SGD", # optimizasyon algoritmasi (Stochastic Gradient Descent), alternatif olarak "Adam" da kullanilabilir
      weight_decay = 0.0005, # agirlik cezasi, overfittingi engellemek icin kullanilir
      momentum = 0.935, # SGD momentum degeri, optimizasyonun daha stabil ve hizli olmasini saglar
      patience = 50, # early stopping icin sabir degeri, modelin gelismedigi durumlarda egitimi durdurur
      workers = 2, # data loader worker sayisi, veri yukleme ve on isleme islemlerini hizlandirir
      device = "cuda:0" if torch.cuda.is_available() else "cpu", # egitim icin cihaz secimi, GPU varsa kullanilir, yoksa CPU kullanilir
      save = True, # modelleri kaydet, egitim sonunda en iyi modeli kaydeder
      save_period = 1, # kac epoch'ta bir kayit yapilacagi,
      val = True, # her epoch sonucunda validation yap, modelin performansini izlemek icin kullanilir
      verbose = True # egitim sureci terminale detayli bir sekilde yazdirilir
    )

if __name__ == "__main__":
    freeze_support()   # Windows için güvenli
    main()



r"""

Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
 1/2         0G     0.8841      4.049      1.269         41        640: 14% ━╸────────── 20/147 4.0s/it 1:26<8:33

 box_loss : 0.1 - 0.3 arasi yeterli
 cls_loss : 1'in altina inmeli
 dfl_loss : 0.5 - 1 civarinda olmasi yeterli


 RESULT :

Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 3.3s/it 23.0s
all        219        260      0.358      0.455      0.375      0.287

Images : toplamda 217 tane dogrulama gorseli gelmis
Instances : 258 tane nesne tespit edilmis
Box(P) : tespit edilen kutularin dogrulugu. 0.76 ise modelin tespit ettigi kutularin %76'sinin gercek kutularla iyi eslestigi anlamina gelir
R : recall, modelin gercek pozitifleri ne kadar dogru tahmin ettigini gosterir, 0.455 degeri modelin gercek pozitiflerin %45.5'ini dogru tahmin ettigini gosterir
mAP50 : mean Average Precision at IoU 0.5, modelin genel performansini gosterir, 0.375 degeri modelin genel olarak %37.5 dogru tahmin yaptigini gosterir
mAP50-95 : mean Average Precision at IoU 0.5 to 0.95, modelin genel performansini daha zorlu kosullarda gosterir, 0.287 degeri modelin genel olarak %28.7 dogru tahmin yaptigini gosterir


Box(P)  ≥ ~0.7-8 , > 0.9 cok iyi  
R  ≥ ~0.6-7 makul , > 0.8-9 cok guclu  
mAP50  ≥ ~0.8 , cogu iyi modelde 0.9 a yakin olur  
mAP50-95  ≥ ~0.6–0.7  
:≥ 0.5–0.6: temel/düşük seviyede kullanışlı,

≥ 0.7: iyi,

≥ 0.8: çok iyi,

≥ 0.9: üst düzey/durable SOTA performans.

Ultralytics 8.4.12  Python-3.10.19 torch-2.10.0+cpu CPU (Intel Core i5-10300H 2.50GHz)
Model summary (fused): 73 layers, 3,007,598 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 2.4s/it 16.8s
                   all        219        260      0.766      0.722      0.795      0.639
                  hump         22         22      0.656      0.818      0.765      0.617
              no entry         30         34       0.68        0.5      0.589      0.474
         no overtaking         21         21      0.904      0.429       0.79      0.705
           no stopping         30         30      0.527        0.7       0.67      0.561
             no u turn         20         21      0.406      0.762      0.727      0.605
               parking         21         32      0.868      0.781       0.86      0.651
              roadwork         22         26      0.798      0.759       0.75      0.508
            roundabout         22         23      0.995      0.696      0.927      0.722
        speed limit 40         29         30      0.831      0.817      0.872      0.743
                  stop         20         21          1      0.958      0.995      0.801
Speed: 1.2ms preprocess, 63.4ms inference, 0.0ms loss, 4.4ms postprocess per image
Results saved to C:\Users\onurb\Desktop\TEKNOFEST\Grnt_ileme_BTK\YOLO_trafik_levhasi_tespiti\runs\detect\traffic-sign-model

"""


