from ultralytics import YOLO
import torch
import cv2

# modeli yukle
model = YOLO("runs/detect/traffic-sign-model4/weights/best.pt") # train.py ile egitilen modelin en iyi hali olan best.pt dosyasini yukleyelim

# test edilen gorselin yuklenmesi
image_path = "test3.jpg" # test etmek istedigimiz gorselin yolu
image = cv2.imread(image_path) # gorseli opencv ile yukleyelim

# image tahmini
results = model(image_path)[0]
print(results) # tahmin sonucunu yazdiralim

# kutu cizimi
for box in results.boxes:

    # koordinatlarr
    x1, y1, x2, y2 = map(int, box.xyxy[0]) # kose koordinatlar
    cls_id = int(box.cls[0]) # classification id
    confidence = float(box.conf[0]) # guven skoru
    label = f"{model.names[cls_id]} conf: {confidence:.2f}" # detection etiketi ve guven skoru

    # kutu cizimi
    cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2) # en sondaki 2 parametresi kutu kalinligi, renk (0,255,0) yesil
    
    # etiket image uzerine ekle
    cv2.putText(image,label, (x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2) # etiket yazisi, konum, font, boyut, renk ve kalinlik

# gorseli goster
cv2.imshow("Prediction", image)
cv2.waitKey(0) # herhangi bir tusa basilana kadar bekle
cv2.destroyAllWindows()

# kaydet
cv2.imwrite("predicton_result.jpg", image) # tahmin sonucunu yeni bir gorsel olarak kaydetelim



