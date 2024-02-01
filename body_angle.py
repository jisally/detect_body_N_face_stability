import json
import numpy as np

def calculate_angle(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_theta = dot_product / (norm_a * norm_b)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)  # 각도를 라디안에서 도로 변환

# 법선 벡터들의 json 파일을 불러옴
with open('pose_landmark/body_normal_line.json', 'r') as f:
    planes = json.load(f)

# 각도를 계산해서 저장할 리스트를 생성
body_angle_up_down = []
body_angle_right_left = []

# 각 프레임에서 법선 벡터의 방향을 계산
for plane in planes:
    if plane is not None:
        # 법선 벡터가 위아래로 움직일 때의 각도
        up_down_angle = calculate_angle(plane, [0, 1, 0])

        body_angle_up_down.append(up_down_angle)

        # 법선 벡터가 좌우로 움직일 때의 각도
        left_right_angle = calculate_angle(plane, [1, 0, 0])

        body_angle_right_left.append(left_right_angle)
    else:
        # 벡터가 None인 프레임에서는 각도를 None으로 저장
        body_angle_up_down.append(None)
        body_angle_right_left.append(None)

# 계산한 각도를 json 파일로 저장
with open('pose_landmark/body_angle_up_down.json', 'w') as f:
    json.dump(body_angle_up_down, f)
with open('pose_landmark/body_angle_right_left.json', 'w') as f:
    json.dump(body_angle_right_left, f)
