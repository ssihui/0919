import tkinter as tk
from tkinter import Toplevel, messagebox, filedialog
import cv2
import json
from ultralytics import YOLO, solutions
from PIL import Image, ImageTk
import os
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import logging
import pandas as pd

def save_to_csv():
    csv_file_path = 'D:\\cal\\user_data.csv'
    try:
        name = name_entry.get().strip()  # 清理输入
        age = age_entry.get().strip()
        height = height_entry.get().strip()
        weight = weight_entry.get().strip()
        gender = gender_var.get().strip()
        activity_level = activity_level_var.get().strip()

        tdee = tdee_var.get().strip()
        result1 = result1_var.get().strip()
        cost = cost_var.get().strip()
        less = less_var.get().strip()

        file_exists = os.path.isfile(csv_file_path)

        # 使用 'utf-8' 编码打开 CSV 文件
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Timestamp', 'Name', 'Age', 'Height (cm)', 'Weight (kg)', 'Gender', 'Activity Level', 'TDEE (kcal)', 'Activity1 Result (kcal)', 'Food Cost (kcal)', 'Remaining (kcal)'])
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), name, age, height, weight, gender, activity_level, tdee, result1, cost, less])

        messagebox.showinfo("成功", "數據已成功保存到CSV文件")
        logging.info("用户数据已成功保存到 CSV 文件")

    except Exception as e:
        messagebox.showerror("錯誤", f"保存CSV文件時發生錯誤: {e}")
        logging.error(f"保存CSV文件時發生錯誤: {e}")


# 按钮点击事件处理函数
def cal_button_clicked_1():
    try:
        video1 = video1_entry.get()
        result1, total = calculate_activities(video1, video_canvas)

        result1_var.set(result1)
        total_var.set(total)

         # 计算食物并获取相关数据
        calculate_food()  # 确保在计算食物时已经保存数据

        # 调用保存到 CSV 的函数
        save_to_csv()
    except Exception as e:
        messagebox.showerror("錯誤", f"计算时发生错误: {e}")
        logging.error(f"计算时发生错误: {e}")



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
video_canvas = tk.Canvas(group_videos, width=420, height=280, bg='#FDF7F7')
video_canvas.grid(row=1, column=0)

# 播放固定视频
fixed_video_path_1 = 'C:\\Users\\User\\Downloads\\cal\\count_yolov8_v1.avi'  # 替换为你的固定视频路径
tk.Button(group_videos, text="Play Fixed Video 1", command=lambda: play_fixed_video(fixed_video_path_1, video_canvas), bg='#629677', fg='#000000', font=('Gabriola', 9, 'bold')).grid(row=2, column=0, sticky='e')


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
    with open('D:\\cal\\food_options.json', 'r', encoding='utf-8') as file:
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

# 更新主窗口的界面，加入两个标签用于展示video1、video2的YOLOv8影片
# 该函数用于更新视频帧，持续读取并显示视频的每一帧

def play_fixed_video(video_path, canvas):
    if not os.path.exists(video_path):
        messagebox.showerror("錯誤", f"無法找到影片文件：{video_path}")
        return
    # 打开固定路径的视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return

        # Get the frame size
        height, width = frame.shape[:2]

        # Update the canvas size to match the video frame size
        canvas.config(width=width, height=height)
        # 转换帧为 Tkinter 可用的格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo_image = ImageTk.PhotoImage(image)

        # Update the canvas with the new image
        canvas.create_image(0, 0, image=photo_image, anchor=tk.NW)
        # Keep a reference to the image to prevent garbage collection
        canvas.image = photo_image

        # 如果超过最大递归次数则停止递归
        if canvas.after_id:
            root.after_cancel(canvas.after_id)
        
        # Schedule the next frame update
        canvas.after_id = canvas.after(30, update_frame)  # Adjust the delay as needed

    update_frame()

# 计算活动卡路里和总卡路里
def calculate_activities(video1, video_canvas):
    # Set default results for each activity
    result1 = 0.0

    def calculate_1(video1, video_canvas):
        if video1:
            cap = cv2.VideoCapture(video1)
            if not cap.isOpened():
                print("Error opening video file")
                return 0.0
            
            w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
            video_writer = cv2.VideoWriter("count_yolov8_v1.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) 

            def calculate_angle(a, b, c):
                a = np.array(a)  # First
                b = np.array(b)  # Mid
                c = np.array(c)  # End
    
                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)
    
                if angle > 180.0:
                    angle = 360 - angle
                return angle

            # Curl counter variables
            counter = 0 
            stage = None

            # Setup mediapipe instance
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                def update_frame():
                    nonlocal counter, stage
                    ret, frame = cap.read()
                    if not ret:
                        cap.release()
                        video_writer.release()
                        return 
                    
                    # Continue with processing...
                    # Frame processing code here

                update_frame()

            result1 = counter * 0.4
            return result1 

        else:
            result1 = 0.0

    # 先进行活动的计算
    result1 = calculate_1(video1, video_canvas)
    
    # 然后再进行控件的配置，避免重复调用
    if 'group0' in globals():
        group0.config(width=25, height=9)

    total = result1
    return round(result1, 2), round(total, 2)
     # 处理活动3和消耗热量
'''
    def calculate_activity3_and_burn(activity3_text, burn3_text):
        # 获取用户输入的运动项
        #activity3 = activity3_entry.get("1.0", "end-1c")  # 获取从第1行第0个字符开始到末尾的输入内容，去除最后的换行符
        #burn3 = burn3_entry.get("1.0", "end-1c")  # 获取对应的消耗热量

        # 分行处理输入数据
        activity_lines = [line.strip() for line in activity3_text.splitlines() if line.strip()]
        burn_lines = [line.strip() for line in burn3_text.splitlines() if line.strip()]

        # 检查运动项和消耗热量的数量是否一致
        if len(activity_lines) != len(burn_lines):
            tk.messagebox.showerror("Error", "運動項目和消耗熱量的數量不匹配！")
            return 0.0
    
        burn_values = []

        for burn in burn_lines:
            try:
                burn_value = float(burn)  # 转换消耗热量为浮点数
                burn_values.append(burn_value)
            except ValueError:
                print(f"無效的熱量數值: {burn}")

        # 计算总消耗热量
        result3  = sum(burn_values) if burn_values else 0.0

        # 显示计算结果
        print(f" {result3 }  kcal")
        #tk.messagebox.showinfo("計算結果", f"總消耗熱量: {result3 } kcal")
        return result3
    '''

    


def cal_button_clicked():
    video1 = video1_entry.get()
    '''
    activity3_text = activity3_entry.get("1.0", tk.END).strip()
    burn3_text = burn3_entry.get("1.0", tk.END).strip()
    '''
    result1,  total = calculate_activities( video1,  video_canvas)
    
    result1_var.set(result1)

    total_var.set(total)

def browse_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    entry.delete(0, tk.END)
    entry.insert(0, file_path)




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
tk.Label(group6, text="將運動影片及記錄上傳進行熱量追蹤", bg='#99EEBB', fg='#000000', padx=30, font=('Gabriola', 9, 'bold')).grid(row=1, column=0, sticky='w')
tk.Label(group6, text="運動項目:", bg='#99EEBB', fg='#000000', padx=30, font=('Gabriola', 12, 'bold')).grid(row=2, column=0, sticky='w')
tk.Label(group6, text="消耗熱量(kcal):", bg='#99EEBB', fg='#000000', padx=30, font=('Gabriola', 12, 'bold')).grid(row=2, column=1, sticky='w')
tk.Label(group6, text="影片上傳:", bg='#99EEBB', fg='#000000', padx=30, font=('Gabriola', 12, 'bold')).grid(row=2, column=2, sticky='w')


tk.Label(group6, text="運動1:", bg='#99EEBB', fg='#000000', padx=30, font=('Gabriola', 9, 'bold')).grid(row=3, column=0, sticky='w')
video1_entry = tk.Entry(group6)
video1_entry.grid(row=4, column=2)
tk.Button(group6, text="上傳影片", command=lambda: browse_file(video1_entry), bg = '#99EEBB',fg = '#000000', font=('Gabriola',9,'bold')).grid(row=3, column=2)
'''
tk.Label(group6, text='運動2:\n自行輸入已完成運動及卡路里', bg='#99EEBB', fg='#000000', padx=30, font=('Gabriola', 9, 'bold'), justify = 'left').grid(row=5, column=0, sticky='w')
activity3_entry = tk.Text(group6, height=2, width=12)  # 创建一个文本框用于输入运动项，每行一个
activity3_entry.grid(row=6, column=0)

burn3_entry = tk.Text(group6, height=2, width=12)  # 创建一个文本框用于输入每项运动的消耗热量，每行一个
burn3_entry.grid(row=6, column=1)
'''
result1_var = tk.StringVar()
tk.Entry(group6, textvariable=result1_var).grid(row=4, column=1)

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