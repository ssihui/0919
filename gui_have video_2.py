import tkinter as tk
from tkinter import Toplevel, messagebox, filedialog
import cv2
import json
from ultralytics import YOLO, solutions
from PIL import Image, ImageTk
import os
import mediapipe as mp
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import logging
import pandas as pd
import itertools
import copy
import keras
from tqdm import tqdm

print("TensorFlow version:", tf.__version__)

# Try importing Keras separately

print("Keras version:", keras.__version__)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

model_path = 'my_lstm_model_1.h5' #模型1
model = load_model(model_path)

def save_to_csv():
    csv_file_path = 'D:\\cal\\user_data.csv'

    # 创建一个字典并填充数据
    data = {
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Name': [name_entry.get().strip()],
        'Age': [age_entry.get().strip()],
        'Height (cm)': [height_entry.get().strip()],
        'Weight (kg)': [weight_entry.get().strip()],
        'Gender': [gender_var.get().strip()],
        'Activity Level': [activity_level_var.get().strip()],
        'TDEE (kcal)': [tdee_var.get().strip()],
        'Activity1 Result (kcal)': [result1_var.get().strip()],
        'Food Cost (kcal)': [cost_var.get().strip()],
        'Remaining (kcal)': [less_var.get().strip()],
        'Selected Meals': [', '.join(selected_meals)],  # 假设 selected_meals 是全局变量
        'Selected Drinks': [', '.join(selected_drinks)]  # 假设 selected_drinks 是全局变量
    }

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)

    try:
        # 如果文件存在，追加数据；否则创建新文件
        if os.path.isfile(csv_file_path):
            df.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file_path, mode='w', header=True, index=False)

        messagebox.showinfo("成功", "數據已成功保存到CSV文件")
        logging.info("用户数据已成功保存到 CSV 文件")

    except Exception as e:
        messagebox.showerror("錯誤", f"保存CSV文件時發生錯誤: {e}")
        logging.error(f"保存CSV文件時發生錯誤: {e}")


# 创建主窗口
root = tk.Tk()
root.title("卡路里計算器")
root.configure(background='#FDF7F7')

width = 1700
height = 950
left = 8
top = 3
root.geometry(f'{width}x{height}+{left}+{top}')

# 创建用于显示视频的 LabelFrame 和 Canvas
group_videos = tk.LabelFrame(root, padx=20, background='#99EEBB', fg='#000000', font=('Gabriola', 9, 'bold'), bd=2)
group_videos.grid(row=1, column=2, sticky='nsew')  # 布局，放置在网格的第2行，第0列
group_videos.grid_propagate(False)
group_videos.config(width=30, height=5)
tk.Label(group_videos, text='影像回顧', padx=20, bg='#99EEBB', fg='#000000', font=('Gabriola', 18, 'bold'), bd=2).grid(row=0, column=0, sticky='w')

# 创建 Canvas 用于显示视频
video_canvas = tk.Canvas(group_videos, width=450, height=300, bg='#FDF7F7')
video_canvas.grid(row=1, column=0)




def show_chart():
    # 创建新窗口
    chart_window = Toplevel(root)
    chart_window.title("圖表視窗")

    # 读取 CSV 文件
    csv_file_path = 'D:\\cal\\user_data.csv'
    
    try:
        # 使用 pandas 读取 CSV 文件
        data = pd.read_csv(csv_file_path)

        # 提取需要绘制的列（这里假设我们要绘制 TDEE 和 Activity1 Result 的关系）
        x = data['TDEE (kcal)']  # X 轴数据
        y1 = data['Activity1 Result (kcal)']  
        y2 = data['Food Cost (kcal)']  # 假设我们也想绘制 Food Cost
        
        # 创建子图
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # 2 行 1 列的子图
        
        # 绘制第一个图
        axs[0].plot(x, y1, marker='o', label='Activity1 Result vs TDEE', color='blue')
        axs[0].set_title('Activity1 Result vs TDEE')
        axs[0].set_xlabel('TDEE (kcal)')
        axs[0].set_ylabel('Activity1 Result (kcal)')
        axs[0].legend()

        # 绘制第二个图
        axs[1].plot(x, y2, marker='x', label='Food Cost vs TDEE', color='green')
        axs[1].set_title('Food Cost vs TDEE')
        axs[1].set_xlabel('TDEE (kcal)')
        axs[1].set_ylabel('Food Cost (kcal)')
        axs[1].legend()

        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        messagebox.showerror("錯誤", f"讀取CSV文件時發生錯誤: {e}")

# 读取 JSON 文件
def load_food_options():
    with open('food_options.json', 'r', encoding='utf-8') as file:
        return json.load(file)

food_data = load_food_options()

# 计算BMR和TDEE的函数
def calculate_bmr_tdee():
    try:
        # 获取用户输入的年龄、身高和体重
        age = float(age_entry.get())
        height = float(height_entry.get())
        weight = float(weight_entry.get())
        gender = gender_var.get()
        activity_level = activity_level_var.get()

        # 计算BMR（基础代谢率）
        if gender == '男':
            bmr = (13.7 * weight) + (5 * height) - (6.8 * age) + 66
        else:
            bmr = (9.6 * weight) + (1.8 * height) - (4.7 * age) + 655

        # 活动水平的系数
        activity_multipliers = {
            "無活動：久坐": 1.2,
            "輕量活動：每周運動1-3天": 1.375,
            "中度活動量：每周運動3-5天": 1.55,
            "高度活動量：每周運動6-7天": 1.725,
            "非常高度活動量": 1.9
        }

                # 确保活动水平被正确选择
        if activity_level not in activity_multipliers:
            raise ValueError("請選擇有效的活動水平")
        
        # 计算TDEE（总每日能量消耗）
        tdee = bmr * activity_multipliers.get(activity_level, 1.9)

        # 显示计算结果
        tdee_var.set(f"{tdee:.2f} kcal")
        root.update()
        group2.update()

    except ValueError:
        # 错误处理：输入值无效
        messagebox.showerror("錯誤", "請輸入有效的數字")

# Activity calculation function




label_dict = {
    0: "push-up",
    1: "sit-up",
    2: "squat",
    3: "pull-up",
    4: "run",
    5: "jump",
    9: "rest"
}

def relative_landmark(landmark_list):
    """计算归一化的相对位置，用于运动预测模型"""
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

# 更新标签以显示运动信息
def update_exercise_info_label(counters, total_calories):
    print(f"Counters: {counters}")
    try:
        total_calories = sum(calculate_calories(activity_type, count) for activity_type, count in counters.items())
    except Exception as e:
        print(f"Error calculating total calories: {e}")

    info_text = "Exercises:\n"
    for activity, count in counters.items():
        if count > 0:
            try:
                activity_type = list(label_dict.keys())[list(label_dict.values()).index(activity)]  
            except ValueError as e:
                print(f"Error retrieving activity type for {activity}: {e}")
                continue

dframe  = 1
sumframe = 30 
def play_video_with_landmarks_and_calories(video1, model, dframe = dframe   , sumframe =sumframe   ):
    """通过深度学习模型预测运动类型，计算动作次数并根据运动类型计算卡路里"""
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cap = cv2.VideoCapture(video1)

        if not cap.isOpened():
            print(f"无法打开视频文件: {video1}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        relative_landmark_list_total = []
        frame_count = 0 
        predicted_class = 9 
        predicted_class_count = {}  # 统计每个类别的预测次数
        predicted_class_max = ""  # 初始化最大预测次数的类别
        predicted_class_max_count = 0  # 初始化最大预测次数
        unit_frame_count = 0  # 初始化 30 帧计数器
        total_calories = 0
        #counter = 0  # 动作计数器
        #stage = None  # 动作阶段


        # 初始化计数器
        counters = {class_name: 0 for class_name in label_dict.values()}

        # 创建输出目录
        output_dir = 'output_images'
        os.makedirs(output_dir, exist_ok=True)

         # 创建CSV文件
        csv_path = 'output_data.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['frame', 'second', 'predicted_class_max', 'pushup', 'sit-up', 'squat', 'pullup', 'run', 'jump', 'rest', 'totalcounter'])

        pbar = tqdm(total=total_frames, desc="Processing frames")


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # 检测到关键点时进行处理
            if results.pose_landmarks and hasattr(results.pose_landmarks, 'landmark'):
                # 绘制骨骼关键点
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                                
                post_landmark_list = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
                relative_landmark_list = relative_landmark(post_landmark_list)
                relative_landmark_list_total.append(relative_landmark_list)

                # 每30帧预测一次运动类型
                if len(relative_landmark_list_total) >= dframe:
                    input_data = np.array(relative_landmark_list_total[-dframe:]).reshape(1, dframe, -1)
                    predictions = model.predict(input_data, verbose=0)
                    predicted_class = np.argmax(predictions, axis=-1)[0]
                    predicted_class_name = label_dict.get(predicted_class, "未知")
                    
                    print(f"预测类别: {predicted_class_name}")

                    # 更新预测次数统计
                    predicted_class_count[predicted_class_name] = predicted_class_count.get(predicted_class_name, 0) + 1

                    # 在画面上显示预测结果
                    cv2.putText(frame, f'Predicting: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 显示预测概率
                    for i, (class_name, prob) in enumerate(zip(label_dict.values(), predictions[0])):
                        cv2.putText(frame, f"{class_name}: {prob:.2f}", (10, 60 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # 统计 sumframe 帧
                    unit_frame_count += 1
                    if unit_frame_count == sumframe:
                        predicted_class_max = max(predicted_class_count, key=predicted_class_count.get)
                        predicted_class_max_count = predicted_class_count[predicted_class_max]
                        counters[predicted_class_max] += 1

                        # 计算卡路里
                        calories_burned = calculate_calories(predicted_class, predicted_class_max_count)
                        total_calories += calories_burned

                        unit_frame_count = 0
                        predicted_class_count = {}

                    cv2.putText(frame, f"Most predicted: {predicted_class_max}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


                    cv2.putText(frame, f"Most predicted: {predicted_class_max}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 更新运动信息标签
                    update_exercise_info_label(counters, total_calories)



            # 显示帧数和FPS
            frame_count += 1
            second = frame_count / fps
            cv2.putText(frame, f'Frame: {frame_count/2}/{total_frames}', (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'FPS: {fps:.2f}', (frame.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 保存图像
            output_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(output_path, frame)

            # 更新CSV文件
            total_counter = sum(counters.values())
            with open(csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([frame_count, second, predicted_class_max] + list(counters.values()) + [total_counter])

            cv2.imshow('Video with Landmarks', frame)
            

            frame_count += 1
            # 将帧转换为PhotoImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将 BGR 转为 RGB
            frame = cv2.resize(frame, (450, 300))  # 调整大小
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # 在 Canvas 上显示图像
            video_canvas.create_image(0, 0, anchor='nw', image=imgtk)
            video_canvas.imgtk = imgtk  # 保存引用，防止图片被垃圾回收


            # 更新界面
            root.update_idletasks()
            root.update()

            pbar.update(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        pbar.close()


calories_per_rep = {
        0: 0.4,  # 伏地挺身每次0.4大卡
        1: 0.16,  # 仰臥起坐每次0.16大卡
        2: 0.42,  # 深蹲每次0.42大卡
        3: 1.0,  # 拉單槓每次1大卡
        4: 0.03,  # 跑步每步0.03大卡
        5: 0.19,  # 跳每次0.19大卡
        9: 0
    }
def calculate_calories(activity_type, count):
    """根据运动类型和完成的动作次数计算卡路里"""
    print(f"Calculating calories for activity_type: {activity_type}, count: {count}")
    result1 = calories_per_rep.get(activity_type, 0) * count
    return result1


#影片瀏覽位置
def browse_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    entry.delete(0, tk.END)
    entry.insert(0, file_path)


def cal_button_clicked():
    global global_result1, global_total
    try:
        video1 = video1_entry.get()
        global_result1,  global_totall = play_video_with_landmarks_and_calories(video1, model)
    
        result1_var.set(global_result1)
        total_var.set(global_total)

        
    except Exception as e:
        messagebox.showerror("錯誤", f"計算時發生錯誤: {e}")
        logging.error(f"計算時發生錯誤: {e}")

# 食物计算函数
def calculate_food():
    try:
        # 检查 tdee_var 是否有有效的值
        tdee_str = tdee_var.get().strip()
        print(f"DEBUG: TDEE value = '{tdee_str}'")  # 输出 TDEE 的实际内容
        
        # 从 TDEE 字符串中提取数值
        tdee_float = float(tdee_str.split()[0])  # 直接获取第一个元素（即数值）

        total_str = total_var.get()
        total_float = float(total_str)

        selected_meals = [meal for meal, var in meals_vars.items() if var.get()]
        selected_drinks = [drink for drink, var in drink_vars.items() if var.get()]

        cost = sum(food_data['meals'].get(meal, 0) for meal in selected_meals) + sum(food_data['drinks'].get(drink, 0) for drink in selected_drinks)

        total_have = tdee_float + total_float 
        less = total_have - cost
       
        
        cost_var.set(f"{cost:.2f} kcal")
        less_var.set(f"{less:.2f} kcal")
    except ValueError as e:
        messagebox.showerror("錯誤", f"請輸入有效的數字或檢查數值格式: {e}")

# 按钮点击事件处理函数
def cal_button_clicked_1():
    try:
        # 使用全局变量，而不是重新计算
        result1_var.set(global_result1)
        total_var.set(global_total)

         # 计算食物并获取相关数据
        calculate_food()  # 确保在计算食物时已经保存数据

        # 调用保存到 CSV 的函数
        save_to_csv()
    except Exception as e:
        messagebox.showerror("錯誤", f"计算时发生错误: {e}")
        logging.error(f"计算时发生错误: {e}")



#Tkinter 介面


tk.Label(root, text='AI 熱量管理師', background='#E99771',fg = '#000000', font=('Gabriola',25,'bold'),bd = 2  ).grid(row=0, column=0, sticky='nsew')

group0 = tk.Label(root, padx=10, pady=10, background='#E99771',fg = '#000000', font=('Gabriola',20,'bold'),bd = 2 )
group0.grid(row=2, column=0, sticky='nsew')
group0.grid_propagate(False)
group0.config(width=25, height=9)

tk.Label(group0, text='''這是一個卡路里計算器\n
此應用程序將幫助您計算以下內容：\n
- 基礎代謝率 (BMR) 和總能量消耗 (TDEE)\n
- 依據運動和活動計算消耗的卡路里\n
- 根據選擇的食物計算攝取的卡路里\n
請按照以下步驟操作：\n
1. 輸入基本的身體資訊和運動強度，計算您的 BMR 和 TDEE。\n
2. 上傳運動影片或輸入其他運動資料來計算消耗的卡路里。\n
3. 選擇您吃的食物，計算總攝取的卡路里，並查看剩餘的卡路里。\n
運動強度可參考網站:https://reurl.cc/xv6k9V''', 
                        justify = 'left',bg = '#E99771',fg = '#000000', padx=30, font=('Gabriola',9,'bold')).grid(row=0, column=0, sticky='w')

group1= tk.Label(root, padx=20, pady=10, background='#E99771',fg = '#000000', font=('Gabriola',30,'bold') , bd = 2 )
group1.grid(row=1, column=0, sticky='nsew')
group1.grid_propagate(False)
group1.config(width=25, height=5)


tk.Label(group1, text='基本資料', background='#E99771',fg = '#000000', padx=30, font=('Gabriola',18,'bold'), bd = 2 ).grid(row=0, column=0, sticky='w')
tk.Label(group1, text='姓名', background='#E99771',fg = '#000000', padx=30, font=('Gabriola',12,'bold'), bd = 2 ).grid(row=1, column=0, sticky='w')
name_entry = tk.Entry(group1)
name_entry.insert(0, "王小明")  # 预设值
name_entry.grid(row=1, column=1, sticky='w')
tk.Label(group1, text="年齡(歲):", bg = '#E99771',fg = '#000000', padx=30, font=('Gabriola',9,'bold')).grid(row=2, column=0, sticky='w')
age_entry = tk.Entry(group1)
age_entry.insert(0, "25")  # 预设值
age_entry.grid(row=2, column=1, sticky='w')

tk.Label(group1, text="身高(公分):", bg = '#E99771',fg = '#000000', padx=30, font=('Gabriola',9,'bold')).grid(row=3, column=0, sticky='w')
height_entry = tk.Entry(group1)
height_entry.insert(0, "170")  # 预设值
height_entry.grid(row=3, column=1, sticky='w')

tk.Label(group1, text="體重(公斤):", bg = '#E99771',fg = '#000000', padx=30, font=('Gabriola',9,'bold')).grid(row=4, column=0, sticky='w')
weight_entry = tk.Entry(group1)
weight_entry.insert(0, "70")  # 预设值
weight_entry.grid(row=4, column=1, sticky='w')

tk.Label(group1, text="性別:", bg = '#E99771',fg = '#000000', padx=30, font=('Gabriola',9,'bold')).grid(row=5, column=0, sticky='w')
gender_var = tk.StringVar(value="男")
tk.Radiobutton(group1, text="男", bg = '#E99771',fg = '#000000', font= ('Gabriola',9,'bold'), variable=gender_var, value="男",anchor = 'w').grid(row=5, column=1, sticky = 'w')
tk.Radiobutton(group1, text="女", bg = '#E99771',fg = '#000000', font= ('Gabriola',9,'bold'), variable=gender_var, value="女",anchor = 'w').grid(row=5, column=2, sticky = 'w')

tk.Label(group1, text="活動量:", bg = '#E99771',fg = '#000000', padx=30, font=('Gabriola',9,'bold')).grid(row=6, column=0, sticky='w')
activity_level_var = tk.StringVar(value="無活動：久坐")
activity_level_menu = tk.OptionMenu(group1, activity_level_var, "無活動：久坐", "輕量活動：每周運動1-3天", "中度活動量：每周運動3-5天", "高度活動量：每周運動6-7天", "非常高度活動量") 
activity_level_menu.config  (bg = '#E99771',fg = '#000000', font=('Gabriola',9,'bold'))
activity_level_menu.grid(row=6, column=1, sticky='w')
tk.Label(group1, text='       ', background='#E99771',fg = '#000000', padx=30, font=('Gabriola',12,'bold'), bd = 2 ).grid(row=7, column=0, sticky='w')
tk.Button(group1, text="計算 TDEE", bg='#000000', fg='#ffffff', font=('Gabriola', 9, 'bold'), command=calculate_bmr_tdee).grid(row=8, column=0, columnspan=2)


top1= tk.Label(root, background='#FDF7F7',fg = '#000000', font=('Gabriola',25,'bold'), bd = 2  )
top1.grid(row=0, column=1, sticky='nsew')
top1.grid_propagate(False)
top1.config(width=40, height=2)


group2= tk.Label(top1, background='#EBD663',fg = '#000000', font=('Gabriola',23,'bold'), bd = 2 , anchor = 'w' )
group2.grid(row=0, column=0)
group2.grid_propagate(False)
group2.config(width=20, height=2)
tk.Label(group2, text="TDEE", bg='#EBD663', fg='#000000', padx=30, font=('Gabriola', 18, 'bold')).grid(row=0, column=0,  columnspan=2)
tdee_var = tk.StringVar(value="")
tk.Label(group2, textvariable=tdee_var, bg='#EBD663', fg='#000000', padx=15, font=('Gabriola', 20, 'bold')).grid(row=1, column=1, sticky='nsew')

group2_1= tk.Label(top1, text="+", background='#FDF7F7',fg = '#000000', font=('Gabriola',15,'bold'), bd = 2  )
group2_1.grid(row=0, column=1)
group2_1.grid_propagate(False)
group2_1.config(width=1, height=2)

group3= tk.Label(top1, background='#629677',fg = '#000000', font=('Gabriola',23,'bold'), bd = 2 , anchor = 'e')
group3.grid(row=0, column=2)
group3.grid_propagate(False)
group3.config(width=20, height=2)
tk.Label(group3, text="運動消耗熱量", bg='#629677', fg='#000000', padx=30, font=('Gabriola', 18, 'bold')).grid(row=0, column=0, columnspan=2)
total_var = tk.StringVar()
tk.Label(group3, textvariable=total_var, bg='#629677', fg='#000000', font=('Gabriola', 20, 'bold')).grid(row=1, column=1, sticky='nsew')

top2= tk.Label(root, background='#FDF7F7',fg = '#000000', font=('Gabriola',25,'bold'), bd = 2  )
top2.grid(row=0, column=2, sticky='nsew')
top2.grid_propagate(False)
top2.config(width=40, height=2)

group4_2= tk.Label(top1, text="-", background='#FDF7F7',fg = '#000000', font=('Gabriola',20,'bold'), bd = 2  )
group4_2.grid(row=0, column=3)
group4_2.grid_propagate(False)
group4_2.config(width=1, height=2)

group4_3= tk.Label(top2, text="", background='#34AAD1',fg = '#000000', font=('Gabriola',23,'bold'), bd = 2  )
group4_3.grid(row=0, column=0)
group4_3.grid_propagate(False)
group4_3.config(width=1, height=2)

group4= tk.Label(top2, background='#34AAD1',fg = '#000000', font=('Gabriola',23,'bold'), bd = 2  )
group4.grid(row=0, column=1)
group4.grid_propagate(False)
group4.config(width=20, height=2)
tk.Label(group4, text="攝取熱量", bg='#34AAD1', fg='#000000', padx=30, font=('Gabriola', 18, 'bold')).grid(row=0, column=0,  columnspan=2)
cost_var = tk.StringVar(value="")
tk.Label(group4, textvariable=cost_var, bg='#34AAD1', fg='#000000', font=('Gabriola', 20, 'bold')).grid(row=1, column=1, sticky='nsew')

group4_1= tk.Label(top2, text="=", background='#FDF7F7',fg = '#000000', font=('Gabriola',15,'bold'), bd = 2  )
group4_1.grid(row=0, column=2)
group4_1.grid_propagate(False)
group4_1.config(width=1, height=2)

group5= tk.Label(top2, background='#2589BD',fg = '#000000', font=('Gabriola',23,'bold'), bd = 2 )
group5.grid(row=0, column=3)
group5.grid_propagate(False)
group5.config(width=20, height=2)
tk.Label(group5, text="剩餘", bg='#2589BD', fg='#000000', padx=30, font=('Gabriola', 18, 'bold')).grid(row=0, column=0, columnspan=2)
less_var = tk.StringVar(value="")
tk.Label(group5, textvariable=less_var, bg='#2589BD', fg='#000000', font=('Gabriola', 20, 'bold')).grid(row=1, column=1, sticky='nsew')

#運動
group6= tk.LabelFrame(root, padx=20, background='#99EEBB',fg = '#000000', font=('Gabriola',30,'bold'), bd = 2  )
group6.grid(row=1, column=1, sticky='nsew')
group6.grid_propagate(False)
group6.config(width=30, height=5)

tk.Label(group6, text="今日運動:", bg='#99EEBB', fg='#000000', padx=30, font=('Gabriola', 18, 'bold')).grid(row=0, column=0, sticky='w')
tk.Label(group6, text="將運動影片上傳進行分析運動類別及熱量追蹤", bg='#99EEBB', fg='#000000', padx=30, font=('Gabriola', 9, 'bold')).grid(row=1, column=0, sticky='w')

tk.Label(group6, text="影片上傳:", bg='#99EEBB', fg='#000000', padx=30, font=('Gabriola', 12, 'bold')).grid(row=1, column=1, sticky='w')
exercise_info_label = tk.Label(group6, text="", font=("Arial", 12), bg="white")
exercise_info_label.grid(row=2, column=0, sticky='w')


video1_entry = tk.Entry(group6)
video1_entry.grid(row=4, column=2)
tk.Button(group6, text="上傳影片", command=lambda: browse_file(video1_entry), bg = '#99EEBB',fg = '#000000', font=('Gabriola',9,'bold')).grid(row=3, column=2)

result1_var = tk.StringVar()
tk.Entry(textvariable=result1_var).grid(row=6, column=1)

tk.Label(group6, text="    ", bg='#99EEBB', fg='#000000', padx=30, font=('Gabriola', 12, 'bold')).grid(row=7, column=0, sticky='w')
tk.Button(group6, text="計算", command=cal_button_clicked, bg = '#99EEBB',fg = '#000000', font=('Gabriola',9,'bold')).grid(row=8, column=1)

#食物
group7= tk.LabelFrame(root, padx=20, background='#5ED7FF', width=30, height=8,fg = '#000000', font=('Gabriola',30,'bold'), bd = 2 )
group7.grid(row=2, column=1, sticky='nsew')
group7.grid_propagate(False)
group7.config(width=30, height=3)

tk.Label(group7, text="早餐選擇", bg='#5ED7FF', fg='#000000', padx=30, font=('Gabriola', 18, 'bold')).grid(row=0, column=0, sticky='w')
tk.Label(group7, text="幫你計算早餐攝取多少熱量", bg='#5ED7FF', padx=30, fg='#000000', font=('Gabriola', 9, 'bold')).grid(row=1, column=0, sticky='w')

# 主餐部分
group7_1 = tk.LabelFrame(group7, text='主餐', padx=20, pady=10,  background='#32B6FD', fg='#000000', font=('Arial', 9, 'bold'), bd=2)
group7_1.grid(row=2, column=0, padx=20, sticky='w')

meals_vars = {}  # 存储主餐 Checkbutton 的变量
row = 0  # 初始化行号
col = 0  # 初始化列号
for meals in food_data["meals"]:
    var = tk.BooleanVar()
    # 使用grid布局，每行显示7个Checkbutton
    tk.Checkbutton(group7_1, text=meals, variable=var, bg='#32B6FD', fg='#000000', font=('Arial', 9)).grid(row=row, column=col, sticky='w')
    meals_vars[meals] = var
    col += 1  # 列号加1

    # 如果列号达到7，换行，列号归零
    if col == 5:
        col = 0
        row += 1

tk.Label(group7, text="    ", bg='#5ED7FF', fg='#000000', padx=30, font=('Gabriola', 5, 'bold')).grid(row=3, column=0, sticky='w')

# 饮料部分
group7_2 = tk.LabelFrame(group7, text='饮料', padx=20, pady=10, background='#32B6FD', fg='#000000', font=('Arial', 9, 'bold'), bd=2)
group7_2.grid(row=4, column=0, padx=20, sticky='w')

drink_vars = {}
row = 0  # 初始化行号
col = 0  # 初始化列号
for drink in food_data["drinks"]:
    var = tk.BooleanVar()
    # 使用grid布局，每行显示7个Checkbutton
    tk.Checkbutton(group7_2, text=drink, variable=var, bg='#32B6FD', fg='#000000', font=('Arial', 9)).grid(row=row, column=col, sticky='w')
    drink_vars[drink] = var
    col += 1  # 列号加1

    # 如果列号达到7，换行，列号归零
    if col == 4:
        col = 0
        row += 1

tk.Label(group7, text="    ", bg='#5ED7FF', fg='#000000', padx=30, font=('Gabriola', 5, 'bold')).grid(row=5, column=0, sticky='w')
tk.Button(group7, text="Calculate Food",padx=20, pady=10, bg='#32B6FD', fg='#000000', font=('Arial', 9, 'bold'), command=calculate_food).grid(row=6, column=0, columnspan=2)


# 创建一个 LabelFrame 作为 group8
group8 = tk.LabelFrame(root, padx=20, background='#A684C2', fg='#000000', font=('Gabriola', 30, 'bold'), bd=2)
group8.grid(row=2, column=2, sticky='nsew')
group8.grid_propagate(False)  # 防止 group8 自动调整大小
group8.config(width=30, height=4)  # 设置固定的宽度和高度

# 在 group8 中添加 "建議" 标签
suggestion_label = tk.Label(group8, text="建議", padx=20, background='#A684C2', fg='#000000', font=('Gabriola', 18, 'bold'))
suggestion_label.grid(row=0, column=0, sticky='w')

# 在 group8 中创建显示图表的按钮
chart_button = tk.Button(group8, text="顯示圖表", command=show_chart, bg='#654F6F', fg='#000000', font=('Gabriola', 12, 'bold'))
chart_button.grid(row=1, column=0, padx=30, pady=30)  # 使用 padx 和 pady 确保按钮有足够的空间
calculate_button = tk.Button(group8, text="保存資料", command=cal_button_clicked_1)
calculate_button.grid(row=1, column=1)  # 使用 pack() 方法布局，添加适当的间距


root.mainloop()