import json
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

with open('pose_landmark/left_shoulder.json') as f:
    left_shoulder = json.load(f)
with open('pose_landmark/right_shoulder.json') as f:
    right_shoulder = json.load(f)

center = (np.array(left_shoulder[0]) + np.array(right_shoulder[0])) / 2
origin = np.copy(center)
origin[1] += (1 - origin[1]) / 2

planes = []
frames = []
left_shoulder_filtered = []
right_shoulder_filtered = []
for i, (l, r) in enumerate(zip(left_shoulder, right_shoulder)):
    frames.append(i)
    if None in l or None in r:
        planes.append(None)
        left_shoulder_filtered.append(None)
        right_shoulder_filtered.append(None)
        continue
    l = np.array(l)
    r = np.array(r)
    plane = np.cross(l - origin, r - origin)
    plane /= np.linalg.norm(plane)
    planes.append(plane.tolist())
    left_shoulder_filtered.append(l)
    right_shoulder_filtered.append(r)

with open('pose_landmark/body_normal_line.json', 'w') as f:
    json.dump(planes, f)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(i):
    print(f"Processing frame {i} / {len(planes)}")  # 진행 상황 출력
    ax.cla()
    if planes[i] is not None and left_shoulder_filtered[i] is not None and right_shoulder_filtered[i] is not None:
        ax.quiver(origin[0], origin[1], origin[2], planes[i][0], planes[i][1], planes[i][2])
        ax.plot([origin[0], left_shoulder_filtered[i][0]], [origin[1], left_shoulder_filtered[i][1]], [origin[2], left_shoulder_filtered[i][2]], 'r-')
        ax.plot([origin[0], right_shoulder_filtered[i][0]], [origin[1], right_shoulder_filtered[i][1]], [origin[2], right_shoulder_filtered[i][2]], 'r-')
        ax.plot([left_shoulder_filtered[i][0], right_shoulder_filtered[i][0]], [left_shoulder_filtered[i][1], right_shoulder_filtered[i][1]], [left_shoulder_filtered[i][2], right_shoulder_filtered[i][2]], 'r-')

    ax.quiver(0, 0, 0, 1, 0, 0, color='b', alpha=0.5, label='x')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.5, label='y')
    ax.quiver(0, 0, 0, 0, 0, 1, color='r', alpha=0.5, label='z')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([1, 0])
    ax.set_zlim([0, -1])
    ax.legend()
    ax.set_title(f"Frame: {frames[i]}")

ani = FuncAnimation(fig, update, frames=len(planes), interval=200, repeat=False)
plt.show()

# 'origin'을 json 파일로 저장
with open('pose_landmark/origin.json', 'w') as f:
    json.dump(origin.tolist(), f)
