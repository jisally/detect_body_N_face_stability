import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#이것도 수정완료여

# json 파일들이 있는 디렉토리 경로
directory = 'C:/Users/seungyeon0510/Desktop/kist_2024/main/mediapipee'

# 3D plot을 생성합니다.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 디렉토리 내의 모든 json 파일에 대해
for filename in os.listdir(directory):
    if filename.endswith('.json'):  # json 파일만 처리
        filepath = os.path.join(directory, filename)

        # json 파일을 읽고, 데이터를 불러옴
        with open(filepath, 'r') as f:
            data = json.load(f)

        # 첫 번째 프레임의 좌표를 가져옴
        x, y, z = data[0]

        # 3D plot에 찍습니다.
        ax.scatter(x, y, z)

        # 점 위에 랜드마크 이름을 표시합니다.
        ax.text(x, y, z, filename[:-5])

# 각 축의 라벨을 설정합니다.
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# plot을 보여줍니다.
plt.show()
