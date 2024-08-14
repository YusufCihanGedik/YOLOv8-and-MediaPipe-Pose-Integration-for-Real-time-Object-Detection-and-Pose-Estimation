import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from ultralytics import YOLO
import time
import threading
import csv

# Model paths
yolo_model_path = r'C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\model\yolov8x.pt'
mediapipe_model_path = r"C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\model\pose_landmarker_heavy.task"
video_source = r"C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\video_img\KESIM_BASKENT_20240730085000_20240730095909_93482899.mp4"

# Masa Pointleri (orijinal görüntüye göre)
original_points = np.array([
    (728, 860), (610, 992), (514, 1116), (638, 1174), (832, 1206), (1014, 1206), (1058, 1094), (1058, 974), (920, 902)
])

# Kırpma parametreleri
crop_y1, crop_y2 = 640, 1700
crop_x1, crop_x2 = 264, 1400

# Kırpılmış görüntüdeki noktalar için güncelleme
cropped_points = original_points - np.array([crop_x1, crop_y1])

# İstediğiniz belirli landmark ID'leri
desired_landmarks = {15, 16, 17, 18, 19, 20, 21, 22}  # Eller için

# İstediğiniz belirli bağlantılar
desired_connections = [
    (11, 12), (11, 13), (11, 23), (12, 24), (12, 14), 
    (13, 15), (14, 16), (15, 17), (15, 19), (15, 21), 
    (16, 22), (16, 18), (16, 20), (23, 24)
]

# MediaPipe Pose settings
num_poses = 4
min_pose_detection_confidence = 0.7
min_pose_presence_confidence = 0.7
min_tracking_confidence = 0.7

# Load YOLOv8 model
yolo_model = YOLO(yolo_model_path)

# CSV dosyasını oluşturun ve başlıkları yazın
# with open('output.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Time", "In_ROI"])

def detect_objects(img, model):
    results = model(img)
    class_ids = []
    confidences = []
    boxes = []

    for result in results:
        for det in result.boxes:
            xmin, ymin, xmax, ymax = map(int, det.xyxy[0])
            confidence = det.conf[0]
            class_id = int(det.cls[0])
            if class_id == 0:  # Sadece insanlar
                if confidence > 0.7:
                    boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    return boxes, confidences, class_ids

def are_landmarks_in_roi(landmarks, roi_points):
    """Check if any of the given landmarks (e.g., hands) are within the ROI."""
    for landmark in landmarks:
        x, y = int(landmark.x * (crop_x2 - crop_x1)), int(landmark.y * (crop_y2 - crop_y1))  # Kırpılmış görüntü boyutunu ayarlayın
        if cv2.pointPolygonTest(roi_points.astype(np.int32), (x, y), False) >= 0:
            print("asdddddddddddddd",cv2.pointPolygonTest(roi_points.astype(np.int32), (x, y), False))
            return True
    return False

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    in_roi = False

    for pose_landmarks in pose_landmarks_list:
        filtered_landmarks = [pose_landmarks[i] for i in desired_landmarks if i < len(pose_landmarks)]
        
        if are_landmarks_in_roi(filtered_landmarks, cropped_points):
            in_roi = True

        # Belirtilen bağlantılar arasında çizim yap
        for connection in desired_connections:
            start_idx, end_idx = connection
            if start_idx in desired_landmarks and end_idx in desired_landmarks:
                start_point = pose_landmarks[start_idx]
                end_point = pose_landmarks[end_idx]
                start_x, start_y = int(start_point.x * rgb_image.shape[1]), int(start_point.y * rgb_image.shape[0])
                end_x, end_y = int(end_point.x * rgb_image.shape[1]), int(end_point.y * rgb_image.shape[0])
                cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        # Belirtilen landmark'ların üzerine daire çiz
        for idx in desired_landmarks:
            if idx < len(pose_landmarks):
                lm = pose_landmarks[idx]
                cx, cy = int(lm.x * annotated_image.shape[1]), int(lm.y * annotated_image.shape[0])
                cv2.circle(annotated_image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # Kırpılmış görüntü üzerinde poligon çizin

    # Güncellenen annotated_image'i img'ye kopyalayın
    rgb_image[:] = annotated_image[:]

    # CSV dosyasına yazma işlemi
    with open('output.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.time(), in_roi])

    return rgb_image


def process_frame_with_yolo_and_pose(img):
    boxes, confidences, class_ids = detect_objects(img, yolo_model)
    
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        print(f"Box: {x}, {y}, {w}, {h}, Confidence: {confidences[i]}")

        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"Person: {confidences[i]:.2f}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Sadece bu kutu içindeki bölge üzerinde pose estimation yap
        crop_img = img[y:y+h, x:x+w]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        
        base_options = python.BaseOptions(model_asset_path=mediapipe_model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False)

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            detection_result = landmarker.detect(mp_image)
            if detection_result.pose_landmarks:
                crop_img = draw_landmarks_on_image(crop_img, detection_result)
                img[y:y+h, x:x+w] = crop_img

    return img

def main(video_path):
    pTime = 0
    last_time = time.time()

    camera = cv2.VideoCapture(video_path)
    fps = camera.get(cv2.CAP_PROP_FPS)
    print("FPS:", fps)

    def process_frame():
        nonlocal last_time
        while True:
            current_time = time.time()
            if current_time - last_time >= 1:  # Her geçen saniyede bir işlem yap
                success, img = camera.read()
                img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                cv2.polylines(img, [cropped_points], isClosed=True, color=(0, 255, 0), thickness=2)
                
                if not success:
                    print("Image capture failed.")
                    break
                img = process_frame_with_yolo_and_pose(img)
                last_time = current_time
                cv2.imshow("YOLO and MediaPipe Pose", cv2.resize(img, (1000, 700)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    thread = threading.Thread(target=process_frame)
    thread.start()

    while True:
        success, img = camera.read()
        cv2.polylines(img, [cropped_points], isClosed=True, color=(0, 255, 0), thickness=2)

        if not success:
            print("Image capture failed.")
            break

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Original Video", cv2.resize(img, (1000, 700)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(video_source)
