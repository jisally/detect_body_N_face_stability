import cv2
import mediapipe as mp
import numpy as np
import json
from tqdm import tqdm

# Face Mesh 모델 로드
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

mp_drawing = mp.solutions.drawing_utils

left_eye_landmarks = [133, 155, 154, 153, 145, 144, 163, 7, 33, 246, 161, 160, 159, 158, 157, 173]
right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

cap = cv2.VideoCapture('C:/Users/seungyeon0510/Desktop/kist_2024/main/video/output.mp4')

# 눈동자 좌표를 저장할 list 생성
eye_center_tracking = []

# 총 프레임 수 계산
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# tqdm을 활용한 진행률 표시
with tqdm(total=total_frames, ncols=70) as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_img)
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                left_eye_point = None
                right_eye_point = None
                for eye_landmarks in [left_eye_landmarks, right_eye_landmarks]:
                    eye_points = np.array([(int(face_landmarks.landmark[point].x * frame.shape[1]),
                                            int(face_landmarks.landmark[point].y * frame.shape[0])) for point in eye_landmarks])
                    if np.min(eye_points[:, 1]) >= np.max(eye_points[:, 1]) or np.min(eye_points[:, 0]) >= np.max(eye_points[:, 0]):
                        continue
                    
                    eye_image = frame[np.min(eye_points[:, 1]):np.max(eye_points[:, 1]),
                                      np.min(eye_points[:, 0]):np.max(eye_points[:, 0])]
                    
                    if eye_image.size == 0:
                        continue
                    
                    eye_image_gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)

                    _, eye_image_binary = cv2.threshold(eye_image_gray, np.mean(eye_image_gray), 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(eye_image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    max_contour = max(contours, key=cv2.contourArea)
                    
                    M = cv2.moments(max_contour)
                    
                    if M["m00"] != 0:
                        cX = int(float(M["m10"] / M["m00"]) + np.min(eye_points[:, 0]))
                        cY = int(float(M["m01"] / M["m00"]) + np.min(eye_points[:, 1]))

                        # 픽셀 위치를 이미지의 너비와 높이로 나눠서 정규화합니다.
                        cX_normalized = cX / frame.shape[1]
                        cY_normalized = cY / frame.shape[0]
                        
                        cv2.circle(frame, (cX, cY), 2, (0, 255, 255), -1)
                        if eye_landmarks == left_eye_landmarks:
                            left_eye_point = (cX_normalized, cY_normalized)
                        else:
                            right_eye_point = (cX_normalized, cY_normalized)

                # 두 눈의 중심점을 계산하여 eye_center_tracking에 추가
                if left_eye_point is not None and right_eye_point is not None:
                    center_x = (left_eye_point[0] + right_eye_point[0]) / 2.0
                    center_y = (left_eye_point[1] + right_eye_point[1]) / 2.0
                    eye_center_tracking.append((center_x, center_y))
                else:
                    eye_center_tracking.append((None, None))
                
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                
                pbar.update(1)
        else:
            eye_center_tracking.append((None, None))
            pbar.update(1)

cap.release()
cv2.destroyAllWindows()

# 눈동자 중심 좌표를 JSON 파일로 저장
with open('zz/eye_center_tracking.json', 'w') as f:
    json.dump(eye_center_tracking, f)
