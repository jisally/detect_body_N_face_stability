import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# json 파일에서 랜드마크 데이터를 불러옵니다.
with open('body_landmark/left_shoulder.json') as f:
    left_shoulder = json.load(f)
with open('body_landmark/right_shoulder.json') as f:
    right_shoulder = json.load(f)

# 각 프레임에서 두 어깨의 중점을 계산합니다.
centers = []

for l, r in zip(left_shoulder, right_shoulder):
    if None in l or None in r:
        # 누락된 데이터(null)가 있는 경우, 해당 프레임의 중점을 None으로 설정합니다.
        centers.append(None)
    else:
        center = [(l[i] + r[i]) / 2 for i in range(3)]
        centers.append(center)

# 중점 데이터를 json 파일에 저장합니다.
with open('body_landmark/center_shoulder.json', 'w') as f:
    json.dump(centers, f)

# 중점 데이터를 불러옵니다.
with open('body_landmark/center_shoulder.json') as f:
    centers = json.load(f)

# y축의 변화를 시각화합니다.
fig, ax = plt.subplots()

def update(i):
    ax.cla()
    valid_centers = [c for c in centers[:i+1] if c is not None]
    valid_indices = [idx for idx, c in enumerate(centers[:i+1]) if c is not None]
    if valid_centers:
        ax.plot(valid_indices, [c[1] for c in valid_centers], 'bo-')
    ax.set_xlim([0, len(centers)])
    ax.set_ylim([0, 1])
    ax.set_title(f"Frame: {i}")

ani = FuncAnimation(fig, update, frames=len(centers), interval=200, repeat=False)
plt.show()
