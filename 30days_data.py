import random
import json
from datetime import datetime, timedelta

# 定義卡路里數據
calories_per_rep = {
    'pushup': 0.4,  
    'abworkout': 0.16,  
    'squat': 0.42,  
    'pullup': 1.0,  
    'run': 0.03,  
    'jump': 0.19,  
    'rest': 0
}

# 食物和飲料數據
food_data = {
    "meals": {
        "Sandwich_230": 230,
        "Hamburger_333": 333,
        "Bread_320": 320,
        "Hashbrowns_124": 124                       
    },
    "drinks": {
        "Juice_147": 147,
        "Coke_185": 185,
        "Tea_192": 192,
        "Coffee_256": 256
    }
}

# 根據活動生成卡路里數據
def generate_activity_results():
    activities = ['pushup', 'abworkout', 'squat', 'pullup', 'run', 'jump', 'rest']
    result_lines = []
    total_calories = 0
    
    for activity in activities:
        count = random.randint(0, 10)  # 隨機生成0到10次的活動次數
        calories = count * calories_per_rep[activity]  # 根據次數計算卡路里
        result_lines.append(f"{activity.capitalize()}: {count}Time, {calories:.2f} kcal")
        total_calories += calories
    
    return "\n".join(result_lines), total_calories

# 根據食物計算卡路里消耗
def calculate_food_cost(selected_meals, selected_drinks):
    cost = sum(food_data['meals'].get(meal, 0) for meal in selected_meals) + sum(food_data['drinks'].get(drink, 0) for drink in selected_drinks)
    return cost

# 隨機生成 30 筆 JSON 資料
json_data = []

# 初始時間
base_time = datetime.now()

for i in range(30):
    # 隨機生成一個負的時間偏移，使得時間早於當前時間
    date_offset = timedelta(days=-random.randint(0, 30),  # 隨機減少 0-30 天
                            hours=-random.randint(0, 23),  # 隨機減少小時
                            minutes=-random.randint(0, 59),  # 隨機減少分鐘
                            seconds=-random.randint(0, 59))  # 隨機減少秒數
    random_time = base_time + date_offset


    activity_result, activity_total_calories = generate_activity_results()
    
    # 隨機選擇食物和飲料
    selected_meals = random.sample(list(food_data['meals'].keys()), random.randint(1, 2))
    selected_drinks = random.sample(list(food_data['drinks'].keys()), random.randint(1, 2))
    
    food_cost = calculate_food_cost(selected_meals, selected_drinks)
    tdee = 2046.00  # 固定 TDEE 值
    remaining_calories = tdee + activity_total_calories - food_cost
    
    # 構建 JSON 結構
    entry = {
        "Timestamp": random_time.strftime('%Y-%m-%d %H:%M:%S'),  # 注意這裡不再是列表形式，而是字符串
        "Name": "王小明",
        "Age": "25",
        "Height (cm)": "170",
        "Weight (kg)": "70",
        "Gender": "男",
        "Activity Level": "無活動：久坐",
        "TDEE (kcal)": f"{tdee:.2f} kcal",
        "Activity  (kcal)": f"{activity_total_calories:.2f} kcal",
        "Activity1 Result (kcal)": activity_result,
        "Food Cost (kcal)": f"{food_cost:.2f} kcal",
        "Remaining (kcal)": f"{remaining_calories:.2f} kcal",
        "Selected Meals": ', '.join(selected_meals),
        "Selected Drinks": ', '.join(selected_drinks)
    }
    
    json_data.append(entry)

# 將數據按時間排序
json_data_sorted = sorted(json_data, key=lambda x: x["Timestamp"])

# 將數據保存為 JSON 文件
with open('random_user_data_sorted.json', 'w', encoding='utf-8') as f:
    json.dump(json_data_sorted, f, ensure_ascii=False, indent=4)

print("30筆資料已生成並保存到 'random_user_data_sorted.json'")