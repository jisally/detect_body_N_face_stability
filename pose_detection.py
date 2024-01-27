import cv2
import mediapipe as mp
import numpy as np
import json
from tqdm import tqdm

#tnwjd

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection

# 랜드마크의 id와 이름을 매핑하는 딕셔너리에 귀에 해당하는 랜드마크를 추가합니다.
landmarks_names = {
    0: 'nose',
    1: 'left_eye_inner',
    2: 'left_eye',
    3: 'left_eye_outer',
    4: 'right_eye_inner',
    5: 'right_eye',
    6: 'right_eye_outer',
    7: 'left_ear',
    8: 'right_ear',
    11: 'left_shoulder',
    12: 'right_shoulder'
}

def calculate_center(*landmarks):
    x = np.mean([landmark[0] for landmark in landmarks])
    y = np.mean([landmark[1] for landmark in landmarks])
    z = np.mean([landmark[2] for landmark in landmarks])
    return (x, y, z)

cap = cv2.VideoCapture('C:/Users/seungyeon0510/Desktop/kist_2024/main/영상데이터/output.mp4')

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
desired_time = 10
frame_num_to_start = int(desired_time * fps)

# landmark_ids 리스트를 업데이트합니다.
landmark_ids = list(landmarks_names.keys())

# landmarks_data 딕셔너리를 업데이트합니다.
landmarks_data = {id: [] for id in landmark_ids}

head_centers = []
shoulder_centers = []
body_centers = []

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_to_start)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    for frame_num in tqdm(range(frame_num_to_start, frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results_pose = pose.process(image)
        results_face = face_detection.process(image)

        if results_pose.pose_landmarks and results_face.detections:
            image_height, image_width, _ = frame.shape
            for i, landmark in enumerate(results_pose.pose_landmarks.landmark):
                if i in landmark_ids:
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z
                    landmarks_data[i].append((x, y, z))
                    
                    landmark_pixel = mp_drawing._normalized_to_pixel_coordinates(x, y, image_width, image_height)
                    if landmark_pixel:
                        radius = int(5 * (1 - z))
                        cv2.circle(frame, landmark_pixel, radius, (0, 0, 255), -1)

            # 두 귀의 랜드마크가 모두 인식되었는지 확인 후 head_center 계산
            if all(len(landmarks_data[id]) > 0 for id in [7, 8]):
                head_center = calculate_center(
                    landmarks_data[7][-1],
                    landmarks_data[8][-1]
                )
                head_centers.append(head_center)
            else:
                head_centers.append((None, None, None))

            shoulder_center = calculate_center(
                landmarks_data[11][-1],
                landmarks_data[12][-1]
            )
            shoulder_centers.append(shoulder_center)

            body_center = calculate_center(head_center, shoulder_center)
            body_centers.append(body_center)

            # 중심점들을 프레임에 표시합니다.
            if None not in head_center:
                frame = cv2.circle(frame, (int(head_center[0]*image_width), int(head_center[1]*image_height)), 5, (0, 255, 0), -1)
            frame = cv2.circle(frame, (int(shoulder_center[0]*image_width), int(shoulder_center[1]*image_height)), 5, (0, 255, 0), -1)
            frame = cv2.circle(frame, (int(body_center[0]*image_width), int(body_center[1]*image_height)), 5, (0, 255, 0), -1)
                
        else:
            for id in landmark_ids:
                landmarks_data[id].append((None, None, None))
            head_centers.append((None, None, None))
            shoulder_centers.append((None, None, None))
            body_centers.append((None, None, None))

        cv2.imshow('MediaPipe Pose', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()

# landmarks_data, head_centers, shoulder_centers, body_centers를 각각의 JSON 파일로 저장합니다.
for id, data in landmarks_data.items():
    with open(f'C:/Users/seungyeon0510/Desktop/kist_2024/main/mediapipee/{landmarks_names[id]}.json', 'w') as f:
        json.dump(data, f)

with open('C:/Users/seungyeon0510/Desktop/kist_2024/main/mediapipee/head_centers.json', 'w') as f:
    json.dump(head_centers, f)

with open('C:/Users/seungyeon0510/Desktop/kist_2024/main/mediapipee/shoulder_centers.json', 'w') as f:
    json.dump(shoulder_centers, f)

with open('C:/Users/seungyeon0510/Desktop/kist_2024/main/mediapipee/body_centers.json', 'w') as f:
    json.dump(body_centers, f)
