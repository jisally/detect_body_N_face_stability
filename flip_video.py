import cv2

# 영상 파일 열기
cap = cv2.VideoCapture('C:/Users/seungyeon0510/Desktop/kist_2024/main/video/original/[10회차 테이블] 230829_10_SI009D0F_T.mp4')

# 영상의 너비와 높이 구하기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 결과 영상을 저장하기 위한 VideoWriter 객체 생성
out = cv2.VideoWriter('C:/Users/seungyeon0510/Desktop/kist_2024/main/video/flipped_[10회차 테이블] 230829_10_SI009D0F_T.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # 영상 좌우 반전
        flipped_frame = cv2.flip(frame, 1)
        
        # 반전된 영상 쓰기
        out.write(flipped_frame)
    else:
        break

# 자원 해제
cap.release()
out.release()
