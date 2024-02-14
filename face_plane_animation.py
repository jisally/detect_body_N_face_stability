import json
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# json 파일에서 랜드마크 데이터를 불러옵니다.
with open('face_landmark_fin/top_face.json') as f:
    top_face = json.load(f)
with open('face_landmark_fin/right_face.json') as f:
    right_face = json.load(f)
with open('face_landmark_fin/left_face.json') as f:
    left_face = json.load(f)

# 각 프레임에서 세 점을 지나는 평면과 그 평면의 법선 벡터를 계산합니다.
planes = []
frames = []
top_face_filtered = []
right_face_filtered = []
left_face_filtered = []

for i, (t, r, l) in enumerate(zip(top_face, right_face, left_face)):
    if None in t or None in r or None in l:  # 좌표가 [None, None, None]인 경우를 건너뜁니다.
        planes.append(None)
        frames.append(i)
        top_face_filtered.append(None)
        right_face_filtered.append(None)
        left_face_filtered.append(None)
        continue

    t = np.array(t)
    r = np.array(r)
    l = np.array(l)
    plane = np.cross(t - r, l - r)
    plane = -plane  # 법선 벡터의 방향을 반전합니다.
    plane /= np.linalg.norm(plane)  # 평면의 법선 벡터를 정규화합니다.

    planes.append(plane.tolist())  # numpy 배열을 리스트로 변환하여 저장합니다.
    frames.append(i)
    top_face_filtered.append(t)
    right_face_filtered.append(r)
    left_face_filtered.append(l)

# 법선 벡터를 json 파일에 저장합니다.
with open('face_landmark_fin/face_normal_line2.json', 'w') as f:
    json.dump(planes, f)

# 평면을 시각화합니다.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(i):
    ax.cla()

    if planes[i] is not None:  # 법선 벡터가 계산되지 않은 프레임을 건너뜁니다.
        ax.quiver(right_face_filtered[i][0], right_face_filtered[i][1], right_face_filtered[i][2], planes[i][0], planes[i][1], planes[i][2])  # 법선 벡터
        ax.plot([right_face_filtered[i][0], top_face_filtered[i][0]], [right_face_filtered[i][1], top_face_filtered[i][1]], [right_face_filtered[i][2], top_face_filtered[i][2]], 'r-')  # 오른쪽 얼굴에서 위쪽 얼굴로
        ax.plot([right_face_filtered[i][0], left_face_filtered[i][0]], [right_face_filtered[i][1], left_face_filtered[i][1]], [right_face_filtered[i][2], left_face_filtered[i][2]], 'r-')  # 오른쪽 얼굴에서 왼쪽 얼굴로
        ax.plot([top_face_filtered[i][0], left_face_filtered[i][0]], [top_face_filtered[i][1], left_face_filtered[i][1]], [top_face_filtered[i][2], left_face_filtered[i][2]], 'r-')  # 위쪽 얼굴에서 왼쪽 얼굴로
    
    # x, y, z 축을 시각화합니다.
    ax.quiver(0, 0, 0, 1, 0, 0, color='b', alpha=0.5, label='x')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.5, label='y')
    ax.quiver(0, 0, 0, 0, 0, 1, color='r', alpha=0.5, label='z')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([1, 0])  # y축의 방향을 반전시킵니다.
    ax.set_zlim([0, -1])  # z축의 방향을 카메라에서 멀어질수록 위로 설정합니다.
    ax.legend()
    ax.set_title(f"Frame: {frames[i]}")  # 현재 보여주는 법선벡터가 어떤 프레임인지 나타냅니다.

ani = FuncAnimation(fig, update, frames=len(planes), interval=200, repeat=False)
plt.show()
