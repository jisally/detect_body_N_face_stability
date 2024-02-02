#colab에서 작성
'''
!wget https://pjreddie.com/media/files/yolov3.weights
!wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true -O ./yolov3.cfg
!wget https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true -O ./coco.names
'''
import cv2
import numpy as np
from tqdm import tqdm

# YOLO 모델 로드
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# 클래스 이름 로드
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] if isinstance(i, list) else layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 비디오 로드
cap = cv2.VideoCapture('/content/drive/MyDrive/Colab Notebooks/crop_crop_240123_13_SI008L0F_T2.mp4')
_, img = cap.read()
height, width, channels = img.shape

# 원본 영상의 프레임 속도 구하기
fps = cap.get(cv2.CAP_PROP_FPS)

# 총 프레임 수 구하기
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 객체 탐지
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# 사람 객체 정보 저장
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id == 0:  # 'person' 클래스만 탐지
            # 객체 탐지
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # 경계 상자 좌표
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])

# 오른쪽 사람 찾기
boxes = sorted(boxes, key=lambda x:x[0])
right_box = boxes[-1]  # 가장 오른쪽에 있는 사람의 경계 상자

# 오른쪽 사람의 박스 중에서 왼쪽 사람과 가까운 지점의 x좌표 구하기
boundary_x = right_box[0]  # 오른쪽 사람의 경계 상자의 왼쪽 끝 x좌표

# 비디오 작성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/content/drive/MyDrive/Colab Notebooks/child_focus_240123_13_SI008L0F_T2.mp4', fourcc, fps, (boundary_x, height))

# tqdm을 사용하여 진행 상태 바 출력
for i in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()
    if ret==True:
        # 경계선을 기준으로 왼쪽만 추출
        left_half = frame[:, :boundary_x]
        out.write(left_half)
    else:
        break

# 비디오 해제
cap.release()
out.release()
cv2.destroyAllWindows()

'''
이후 영상을 10fps로 변환
ffmpeg -i input.mp4 -r 10 output.mp4
'''
