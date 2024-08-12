# # # import cv2
# # # import mediapipe as mp
# # # import time

# # # mpDraw = mp.solutions.drawing_utils
# # # mpPose = mp.solutions.pose
# # # pose = mpPose.Pose()

# # # cap = cv2.VideoCapture(r'C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\kesim.mp4')
# # # pTime = 0
# # # while True:
# # #     success, img = cap.read()
# # #     img=img[660:931,540:1254]
# # #     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # #     results = pose.process(imgRGB)
# # #     # print(results.pose_landmarks)
# # #     if results.pose_landmarks:
# # #         mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
# # #         for id, lm in enumerate(results.pose_landmarks.landmark):
# # #             h, w, c = img.shape
# # #             print(id, lm)
# # #             cx, cy = int(lm.x * w), int(lm.y * h)
# # #             cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

# # #     cTime = time.time()
# # #     fps = 1 / (cTime - pTime)
# # #     pTime = cTime

# # #     cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
# # #                 (255, 0, 0), 3)

# # #     cv2.imshow("Image", img)
# # #     cv2.waitKey(1)


# import cv2
# import mediapipe as mp
# import time

# mpDraw = mp.solutions.drawing_utils
# mpPose = mp.solutions.pose
# pose = mpPose.Pose()

# cap = cv2.VideoCapture(r'C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\kesim.mp4')
# pTime = 0
# while True:
#     success, img = cap.read()
#     img = img[660:1435,297:1305]
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = pose.process(imgRGB)

#     if results.pose_landmarks:
#         mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
#         for id, lm in enumerate(results.pose_landmarks.landmark):
#             h, w, c = img.shape
#             cx, cy = int(lm.x * w), int(lm.y * h)
            
#             # Örneğin sadece baş ve omuz noktalarını işleme dahil et
#             if id in [0,11,12,13,14,15,16,17,18,19,20,21,22,23,24]:  # Baş (0), sol omuz (5), sağ omuz (2), sol dirsek (11)
#                 cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
#                 print(id, lm)

#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime

#     cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
#                 (255, 0, 0), 3)

#     cv2.imshow("Image", img)
#     cv2.waitKey(1)
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from ultralytics import YOLO
import time

# Model paths
yolo_model_path = r'C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\best.pt'
mediapipe_model_path = r"C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\pose_landmarker_heavy.task"
video_source = r"C:\Users\Gedik\Desktop\ComputerVision\DeepLearning\yolov8-pose\11.mp4"

# MediaPipe Pose settings
num_poses = 4
min_pose_detection_confidence = 0.7
min_pose_presence_confidence = 0.7
min_tracking_confidence = 0.7

# Load YOLOv8 model
yolo_model = YOLO(yolo_model_path)

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
            if class_id == 1:  # Sadece insanlar
                if confidence > 0.7:
                    boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    return boxes, confidences, class_ids

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    desired_landmarks = {0,11,12,13,14,15,16,17,18,19,20,21,22,23,24}

    desired_connections = [
        (11, 12),  # Sol omuz - Sağ omuz
        (11, 13),
        (11, 23),  # Sol omuz - Sol kalça
        (12, 24),  # Sağ omuz - Sağ kalça
        (12, 14),
        (13,15),
        (14,16),
        (15,17),
        (15,19),
        (15,21),
        (16,22),
        (16,18),
        (16,20),
        (23, 24)   # Sol kalça - Sağ kalça
    ]

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z) for landmark in pose_landmarks
        ])
    #     mp.solutions.drawing_utils.draw_landmarks(
    #         annotated_image,
    #         pose_landmarks_proto,
    #         mp.solutions.pose.POSE_CONNECTIONS,
    #         mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    # return annotated_image
        mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                connections=desired_connections,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
    return annotated_image

to_window = None

def process_frame_with_yolo_and_pose(img):
    global to_window
    boxes, confidences, class_ids = detect_objects(img, yolo_model)
    
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        print(f"Box: {x}, {y}, {w}, {h}, Confidence: {confidences[i]}")

        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"Person: {confidences[i]:.2f}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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


def main(video_path, frame_interval=10):
    pTime = 0
    frame_count = 0

    camera = cv2.VideoCapture(video_path)
    fps = camera.get(cv2.CAP_PROP_FPS)
    print("FPS:", fps)

    while True:
        success, img = camera.read()
        if not success:
            print("Image capture failed.")
            break

        if frame_count % frame_interval == 0:
            img = process_frame_with_yolo_and_pose(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("YOLO and MediaPipe Pose", cv2.resize(img, (1920, 1080)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(video_source)
