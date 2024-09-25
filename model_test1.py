import cv2
import os
import itertools
import copy
import csv
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Mediapipe Pose 模块初始化
mp_pose = mp.solutions.pose

# 你的 relative_landmark 函数和 save_landmarks_to_csv 函数...
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

def save_landmarks_to_csv(landmark_list, video_path):
    """将标志点数据保存为 CSV 文件，包含表头"""
    num_landmarks = len(landmark_list[0]) // 3
    
    # 构建表头，只包含33个特征点的x, y, z
    header = []
    for i in range(0,num_landmarks):
        header.extend([f'x{i}', f'y{i}', f'z{i}'])


    # 获取视频文件名并替换扩展名为 .csv
    base_name = os.path.basename(video_path)
    csv_path = os.path.splitext(base_name)[0] + ".csv"

    # 写入 CSV 文件
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)  # 写入表头
        csv_writer.writerows(landmark_list)  # 写入数据

    print(f"数据已保存至 {csv_path}")

    
def relative_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y, base_z = 0, 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y, base_z = landmark_point[0], landmark_point[1], landmark_point[2]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list[index][2] = temp_landmark_list[index][2] - base_z

    shoulder_length_x = abs(temp_landmark_list[11][0] - temp_landmark_list[12][0])

    if shoulder_length_x == 0:
        shoulder_length_x = 1  # 防止除以0

    for idx, relative_point in enumerate(temp_landmark_list):
        temp_landmark_list[idx][0] = temp_landmark_list[idx][0] / shoulder_length_x
        temp_landmark_list[idx][1] = temp_landmark_list[idx][1] / shoulder_length_x
        temp_landmark_list[idx][2] = temp_landmark_list[idx][2] / shoulder_length_x

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    return temp_landmark_list

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def play_video_with_landmarks(video_path, model):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return

        relative_landmark_list_total = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks and hasattr(results.pose_landmarks, 'landmark'):
                post_landmark_list = []
                for value in results.pose_landmarks.landmark:
                    temp_value = [value.x, value.y, value.z]
                    post_landmark_list.append(temp_value)

                relative_landmark_list = relative_landmark(post_landmark_list)
                relative_landmark_list_total.append(relative_landmark_list)

                for i in range(len(relative_landmark_list) // 3):
                    x = int(relative_landmark_list[i * 3] * frame.shape[1])
                    y = int(relative_landmark_list[i * 3 + 1] * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            if len(relative_landmark_list_total) >= 30:
                input_data = np.array(relative_landmark_list_total[-30:]).reshape(1, 30, -1)
                predictions = model.predict(input_data)
                print("预测结果:", predictions)  # 打印原始预测结果
                predicted_class = np.argmax(predictions, axis=-1)[0]
                print("预测类别:", predicted_class)  # 打印预测类别
                cv2.putText(frame, f'Predicted: {predicted_class}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Video with Landmarks', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# 使用示例
video_path = 'D:\\0925\\sets.mp4'  # 替换为你的视频文件路径
model_path = 'my_lstm_model.h5'  # 替换为你的模型路径
model = load_model(model_path)
play_video_with_landmarks(video_path, model)
