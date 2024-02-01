import cv2
import mediapipe as mp
import json
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video_path = 'C:/Users/seungyeon0510/Desktop/kist_2024/main/영상데이터/crop_output2_origin.mp4'
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
    for frame_no in tqdm(range(total_frames), desc="Processing frames"):
        success, image = cap.read()

        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            for name in landmarks.values():
                if results.pose_landmarks:
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        if idx in landmarks and landmarks[idx] == name:
                            landmark_dict[name].append([landmark.x, landmark.y, landmark.z])
                            break
                    else:
                        landmark_dict[name].append([None, None, None])
                else:
                    landmark_dict[name].append([None, None, None])

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    if idx in landmarks:
                        image_hight, image_width, _ = image.shape
                        landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_width, image_hight)
                        cv2.circle(image, landmark_px, 3, (255, 0, 0), 2)

                        landmark_coords = f"{landmarks[idx]}: x: {landmark.x:.2f}, y: {landmark.y:.2f}, z: {landmark.z:.2f}"
                        cv2.putText(image, landmark_coords, (10, 30 + 20 * idx),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 프레임 번호를 화면에 표시
            cv2.putText(image, f'Frame: {frame_no}', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('MediaPipe Pose', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()
r
for name, coordinates in landmark_dict.items():
    with open(f'pose_landmark/{name}.json', 'w') as f:
        json.dump(coordinates, f)
