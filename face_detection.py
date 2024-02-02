from tqdm import tqdm
import cv2
import mediapipe as mp
import json

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Target landmarks and their names
target_landmarks = [10, 150, 379]
landmark_names = ['top_face', 'right_face', 'left_face']
landmark_map = dict(zip(target_landmarks, landmark_names))

# Loading video
cap = cv2.VideoCapture('C:/Users/seungyeon0510/Desktop/kist_2024/main/video/child_focus_240123_13_SI008L0F_T2_10fps.mp4')

# Get total frame count
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Prepare json data storage
landmark_data = {landmark_map[i]: [] for i in target_landmarks}

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
  for _ in tqdm(range(total_frames), desc="Processing video"):
    success, image = cap.read()
    if not success:
      break

    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        for i in target_landmarks:
            landmark = face_landmarks.landmark[i]
            landmark_data[landmark_map[i]].append([landmark.x, landmark.y, landmark.z])
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    else:
        # If no face detected, add null data
        for i in target_landmarks:
            landmark_data[landmark_map[i]].append([None, None, None])

    # Show video with landmarks
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()

# Save landmarks data to json files
for i in target_landmarks:
    with open(f'face_landmark/{landmark_map[i]}.json', 'w') as f:
        json.dump(landmark_data[landmark_map[i]], f)
