
import json
import cv2
import os
from tqdm import tqdm

# json 파일 읽기
with open('head_bow.json', 'r') as f:
    angles = json.load(f)

# 비디오 파일 열기
cap = cv2.VideoCapture('C:/Users/seungyeon0510/Desktop/kist_2024/main/영상데이터/output.mp4')

# plus, minus 폴더 생성
if not os.path.exists('plus'):
    os.makedirs('plus')
if not os.path.exists('minus'):
    os.makedirs('minus')

# tqdm 사용하여 진행률 표시
for i in tqdm(range(len(angles)), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break

    # 각도 값이 'null'이 아닌 경우에만 처리
    if angles[i] is not None:
        # 각도가 양수인 경우 'plus' 폴더에, 음수인 경우 'minus' 폴더에 저장
        if angles[i] > 0:
            cv2.imwrite(f'plus/frame_{i}.png', frame)
        elif angles[i] < 0:
            cv2.imwrite(f'minus/frame_{i}.png', frame)

cap.release()
