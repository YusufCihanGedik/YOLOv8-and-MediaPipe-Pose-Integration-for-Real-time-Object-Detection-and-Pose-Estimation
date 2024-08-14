import cv2
import mediapipe as mp
import time
import numpy as np
import csv

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

video_source = r"C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\video_img\KESIM_BASKENT_20240730085000_20240730095909_93482899.mp4"
cap = cv2.VideoCapture(video_source)

# Masa Pointleri
# points = np.array([
#   (442, 138), (338, 250), (174, 456), (88, 578), (280, 664), (738, 750), (872, 472), (992, 240), (1010, 172)
# ])

points = np.array([
    (728, 860), (610, 992), (514, 1116), (638, 1174), (832, 1206), (1014, 1206), (1058, 1094), (1058, 974), (920, 902)
])

# CSV dosyası için başlıklar
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "In_ROI"])

# İstediğiniz belirli landmark ID'leri (eller için)
hand_landmarks = {15, 16, 17, 18, 19, 20, 21, 22}

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Görüntüyü kırpma
    img = img[640:1700, 264:1400]
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    in_roi = False

    if results.pose_landmarks:
        for id in hand_landmarks:
            lm = results.pose_landmarks.landmark[id]
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            
            # Eğer landmark (el noktası) ROI poligonu içinde ise
            if cv2.pointPolygonTest(points, (cx, cy), False) >= 0:
                in_roi = True
                break

    # Çokgeni çizme
    cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # CSV dosyasına True veya False yazma
    with open('output.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.time(), in_roi])

    # FPS hesaplama ve görüntüleme
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", cv2.resize(img, (1000, 700)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
