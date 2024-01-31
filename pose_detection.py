import cv2
import mediapipe as mp
import json
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video_path = 'C:/Users/seungyeon0510/Desktop/kist_2024/main/영상데이터/flip_output2.mp4'
cap = cv2.VideoCapture(video_path)

landmarks = {
    0: 'nose',
    2: 'left_eye',
    5: 'right_eye',
    7: 'left_ear',
    8: 'right_ear',
    11: 'left_shoulder',
    12: 'right_shoulder'
}

landmark_dict = {name: [] for name in landmarks.values()}

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        if results.pose_landmarks:
            for name in landmarks.values():
                landmark_dict[name].append('null') # 랜드마크가 검출되지 않은 경우 'null' 추가

            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx in landmarks:
                    landmark_dict[landmarks[idx]].pop()  # 'null' 값을 제거하고
                    landmark_dict[landmarks[idx]].append([landmark.x, landmark.y, landmark.z])  # 좌표를 추가

        # 랜드마크를 영상에 표시
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 랜드마크 좌표를 텍스트로 영상에 표시
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx in landmarks:
                    landmark_coords = f"x: {landmark.x:.2f}, y: {landmark.y:.2f}, z: {landmark.z:.2f}"
                    cv2.putText(image, landmark_coords, (10, 30 + 20 * idx),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

for name, coordinates in landmark_dict.items():
    with open(f'pose_landmark2/{name}.json', 'w') as f:
        json.dump(coordinates, f)
