import cv2
import os

# 동영상 파일 경로
input_video_path = '/home/seungseoroh/Desktop/AIProgramming_seungss/test1_soccer/1/KakaoTalk_20231204_001234936_02.webp'

# 이미지를 저장할 폴더 생성
output_folder = '/home/seungseoroh/Desktop/AIProgramming_seungss/test1_soccer/1'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 동영상을 읽기
cap = cv2.VideoCapture(input_video_path)

# 동영상의 FPS (프레임 속도)를 얻기
fps = cap.get(cv2.CAP_PROP_FPS)

# 추출할 프레임 간격 계산
frame_interval = max(1, int(fps / 30))

# 프레임 추출 및 저장
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 지정된 간격마다 프레임을 저장
    if frame_count % frame_interval == 0:
        frame_file = os.path.join(output_folder, f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_file, frame)

    frame_count += 1

# 자원 해제
cap.release()
