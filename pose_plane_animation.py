import json
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# json 파일에서 랜드마크 데이터를 불러옵니다.
with open('pose_landmark/left_shoulder.json') as f:
    left_shoulder = json.load(f)
with open('pose_landmark/right_shoulder.json') as f:
    right_shoulder = json.load(f)

# 첫 번째 프레임에서 두 어깨의 중점을 구합니다.
center = (np.array(left_shoulder[0]) + np.array(right_shoulder[0])) / 2
origin = np.copy(center)
origin[1] += (1 - origin[1]) / 2

# 각 프레임에서 origin과 두 어깨를 연결하는 평면을 계산합니다.
planes = []
frames = []
left_shoulder_filtered = []
right_shoulder_filtered = []
for i, (l, r) in enumerate(zip(left_shoulder, right_shoulder)):
    if None in l or None in r:  # 좌표가 [None, None, None]인 경우를 건너뜁니다.
        planes.append(None)
        continue
    l = np.array(l)
    r = np.array(r)
    plane = np.cross(l - origin, r - origin)
    plane /= np.linalg.norm(plane)  # 평면의 법선 벡터를 정규화합니다.
    planes.append(plane.tolist())  # numpy 배열을 리스트로 변환하여 저장합니다.
    frames.append(i)
    left_shoulder_filtered.append(l)
    right_shoulder_filtered.append(r)

# 법선 벡터를 json 파일에 저장합니다.
with open('pose_landmark/body_normal_line.json', 'w') as f:
    json.dump(planes, f)

# 평면을 시각화합니다.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(i):
    ax.cla()

    if planes[i] is not None:  # 법선 벡터가 계산되지 않은 프레임을 건너뜁니다.
        ax.quiver(origin[0], origin[1], origin[2], planes[i][0], planes[i][1], planes[i][2])  # 법선 벡터
        ax.plot([origin[0], left_shoulder_filtered[i][0]], [origin[1], left_shoulder_filtered[i][1]], [origin[2], left_shoulder_filtered[i][2]], 'r-')  # origin에서 왼쪽 어깨로
        ax.plot([origin[0], right_shoulder_filtered[i][0]], [origin[1], right_shoulder_filtered[i][1]], [origin[2], right_shoulder_filtered[i][2]], 'r-')  # origin에서 오른쪽 어깨로
        ax.plot([left_shoulder_filtered[i][0], right_shoulder_filtered[i][0]], [left_shoulder_filtered[i][1], right_shoulder_filtered[i][1]], [left_shoulder_filtered[i][2], right_shoulder_filtered[i][2]], 'r-')  # 왼쪽 어깨에서 오른쪽 어깨로
    
    # x, y, z 축을 시각화합니다.
    ax.quiver(0, 0, 0, 1, 0, 0, color='b', alpha=0.5, label='x')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.5, label='y')
    ax.quiver(0, 0, 0, 0, 0, 1, color='r', alpha=0.5, label='z')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([1, 0])  # y축의 방향을 반전시킵니다.
    ax.set_zlim([0, -1])  # z축의 방향을 카메라에서 멀어질수록 위로 설정합니다.
    ax.legend()
    ax.set_title(f"Frame: {frames[i]}")  # 현재 보여주는 법선벡터가 어떤 프레임인지 나타냅니다.

ani = FuncAnimation(fig, update, frames=len(planes), interval=200)
plt.show()
