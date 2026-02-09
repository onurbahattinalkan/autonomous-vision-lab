"""
araclarin takibi : yolo kullanalim, training yapmayalim zaten yolo default olarak araclarin tespitini yapabiliyor
"""
from ultralytics import YOLO
import cv2


# veri seti incele

# yolo modeli yukle 
model = YOLO("yolov8n.pt") # yolov8s, m ,l kullanilabilir

# video giris kaynagi
video_path = "IMG_5269.MOV" # video dosyasinin yolu
cap = cv2.VideoCapture(video_path)

# cikis videosunu yazmak icin ayar
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # video genisligi
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # video yuksekligi
fps = cap.get(cv2.CAP_PROP_FPS) # frame per second, 1 saniyedeki kare sayisi
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height)) # cikis videosu ayarlari

# tracking algoritmasi ve testing
while cap.isOpened():
    success, frame = cap.read() # video karelerini oku
    if not success: # video bitti ise okuma basarisiz olduysa donguyu kir
        break

    # yolo ile tracking
    results = model.track(
        frame, # giris goruntusu
        persist=True, # kalici izleme, her karede yeni bir izleme baslatmaz, mevcut izlemeleri devam ettirir. tespit idleri korunur
        conf = 0.3, # güven esigi, tespitlerin kabul edilmesi icin gereken minimum guven puani. 0.3, 30% guvenle tespitleri kabul eder. 1 mukemmel
        iou = 0.5, # Intersection over Union esigi, tespitlerin birbirleriyle ne kadar örtüştüğünü belirler. 0.5, tespitlerin en az %50 örtüşmesi gerektiği anlamına gelir
        tracker = "bytetrack.yaml", # tracking algoritmasi, bytetrack, strongsort, deepsort gibi algoritmalar kullanilabilir.
        #classes = [2] # sadece araclarin tespit edilmesi icin, 2 class id'si araclari temsil eder. yolo modelinde class id'leri genellikle 0-79 arasinda olur, 2 genellikle araclari temsil eder
        )
    
    #kutulari ve idleri ekran uzerine cizdir
    annotated_fram = results[0].plot() # tespit edilen nesnelerin kutularini ve idlerini cizdirir

    # goster ve kaydet
    cv2.imshow("YOLO v8 Tracking", annotated_fram) # tespit edilen nesnelerin kutularini ve idlerini gosterir
    out.write(annotated_fram) # tespit edilen nesnelerin kutularini ve idlerini cikis videosuna kaydeder

    if cv2.waitKey(1) & 0xFF == ord("q"): # q tusuna basildiginda cikis yap
        break
    
cap.release() # video kaynagini serbest birak
out.release() # cikis video yazicisini kapat
cv2.destroyAllWindows() # tum pencereleri kapat



