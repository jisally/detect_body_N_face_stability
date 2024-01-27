import json
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_angle(head_center, shoulder_center):
    dx = head_center[0] - shoulder_center[0]
    dz = head_center[2] - shoulder_center[2]
    if dx == 0 and dz == 0:
        return 0  # 두 점이 같은 경우 각도는 0
    r = np.sqrt(dx**2 + dz**2)
    theta = np.arcsin(dx / r)
    return np.degrees(theta)

# 랜드마크 파일 경로
landmarks_path = 'C:/Users/seungyeon0510/Desktop/kist_2024/main/mediapipee/'

# 각도 계산을 위한 랜드마크 데이터를 불러옵니다.
with open(os.path.join(landmarks_path, 'head_centers.json'), 'r') as f:
    head_centers = json.load(f)
with open(os.path.join(landmarks_path, 'shoulder_centers.json'), 'r') as f:
    shoulder_centers = json.load(f)

# 각 프레임에 대한 각도를 계산합니다.
angles = []
for head_center, shoulder_center in zip(head_centers, shoulder_centers):
    if None in head_center or None in shoulder_center:
        angles.append(None)  # 잘못된 데이터는 None으로 저장합니다.
        continue
    head_center = np.array(head_center)
    shoulder_center = np.array(shoulder_center)
    angle = calculate_angle(head_center, shoulder_center)
    angles.append(angle)

# 계산된 각도를 JSON 파일로 저장합니다.
with open(os.path.join(landmarks_path, 'head_bow.json'), 'w') as f:
    json.dump(angles, f)

# 각도의 분포를 산점도로 시각화합니다.
plt.figure()
plt.scatter(range(len(angles)), angles, alpha=0.6, color='skyblue', edgecolor='black')
plt.title('Head Tilt Angle Over Frames')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.grid(True)
plt.show()
