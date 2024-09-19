import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import mediapipe as mp
import numpy as np
import cv2
import copy
import itertools
import csv 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import json
from ultralytics import YOLO, solutions

def relative_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x,base_y,base_z,  = 0,0,0
    for index,landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x,base_y,base_z =landmark_point[0], landmark_point[1], landmark_point[2]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z
    
    shoulder_length_x = abs(temp_landmark_list[11][0]-temp_landmark_list[12][0])
    
    if shoulder_length_x == 0:
        shoulder_length_x = 1  # 防止除以0

    for idx, relative_point in enumerate(temp_landmark_list):
        temp_landmark_list[idx][0] = temp_landmark_list[idx][0]/shoulder_length_x
        temp_landmark_list[idx][1] = temp_landmark_list[idx][1]/shoulder_length_x
        temp_landmark_list[idx][2] = temp_landmark_list[idx][2]/shoulder_length_x
    
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    print(temp_landmark_list)
    return temp_landmark_list

def save_landmarks_to_csv(landmark_list, output_csv_path):
    """将标志点数据保存为 CSV 文件，包含表头"""
    num_landmarks = len(landmark_list[0]) // 3
    
    # 构建表头，只包含33个特征点的x, y, z
    header = []
    for i in range(0,num_landmarks-1):
        header.extend([f'x{i+1}', f'y{i+1}', f'z{i+1}'])
    
    # 写入 CSV 文件
    with open(output_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)  # 写入表头
        csv_writer.writerows(landmark_list)  # 写入数据

    print(f"数据已保存至 {output_csv_path}")

def get_poselandmark_xyz(video_path):
    with mp_pose.Pose(
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as pose:
    
        cap = cv2.VideoCapture()
        if not cap.open(video_path):
            print(f"无法打开视频文件: {video_path}")
            return None
        
        relative_landmark_list_total = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True :

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                post_landmark_list = []
                if results.pose_landmarks and hasattr(results.pose_landmarks, 'landmark'):
                    for value in results.pose_landmarks.landmark:
                        temp_value = [value.x,value.y,value.z]
                        post_landmark_list.append(temp_value)

                    relative_landmark_list = relative_landmark(post_landmark_list)#正規化
                    relative_landmark_list_total.append(relative_landmark_list)
            else:
                break


    cap.release()
    return relative_landmark_list_total


video_path = "C:\\Users\\User\\Downloads\\cal\\pushup3.mp4"
output_csv_path = "output_landmarks.csv"

# 获取姿态标志点
landmarks = get_poselandmark_xyz(video_path)

# 保存为 CSV 文件
if landmarks:
    save_landmarks_to_csv(landmarks, output_csv_path)   