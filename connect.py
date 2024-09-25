import pandas as pd
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras import optimizers
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input

# Step 1: 定義要搜尋的基礎資料夾路徑
base_dir = 'D:\\0925'

# Step 2: 搜尋所有output資料夾中的csv檔案
csv_files = glob.glob(os.path.join(base_dir, '**', '*.csv'), recursive=True)

# Step 3: 根據檔案路徑中的資料夾名稱判斷label
label_list = []
for file_path in csv_files:
    if '0-output' in file_path:
        label_list.append(0)
    elif '1-output' in file_path:
        label_list.append(1)
    elif '2-output' in file_path:
        label_list.append(2)
    elif '3-output' in file_path:
        label_list.append(3)
    elif '4-output' in file_path:
        label_list.append(4)
    elif '5-output' in file_path:
        label_list.append(5)
    else:
        label_list.append(None)  # 若無法識別則設置為None

# Step 4: 建立DataFrame
data = {
    'label': label_list,
    'csv_paths': [os.path.relpath(path, base_dir) for path in csv_files]  # 轉換為相對路徑
}

df = pd.DataFrame(data)

# Step 5: 檢查結果
print(df)

# Step 6: 將DataFrame存成CSV檔案並保留索引
output_csv_path = 'label_csv.csv'  # 定義輸出CSV檔案的路徑和名稱
df.to_csv(output_csv_path, index=True)  # 保存索引列

print(f"DataFrame 已經存成 CSV 檔案並保留索引，路徑為: {'D:\\0925'}")

num_samples = 15
frame_length = 50

X = []
Y = []

df = df.dropna(subset=['label'])

for item_label, item_csvpath in df.values:
    
    data = pd.read_csv(item_csvpath)
    data_label = item_label

    if len(data) < frame_length:
        continue
    else:

        data_1 = data.iloc[:int(len(data)*0.2)]
        selected_indices_1 = np.random.choice(data_1.index, size = num_samples, replace = False)
        selected_data_1 = data_1.loc[selected_indices_1]
        sorted_data_1 = selected_data_1.sort_index()

        data_2 = data.iloc[int(len(data)*0.2)+1:int(len(data)*0.4)]
        selected_indices_2 = np.random.choice(data_2.index, size = num_samples, replace = False)
        selected_data_2 = data_2.loc[selected_indices_2]
        sorted_data_2= selected_data_2.sort_index()

        data_3 = data.iloc[-int(len(data)*0.4):]
        selected_indices_3 = np.random.choice(data_3.index, size = num_samples, replace = False)
        selected_data_3 = data_3.loc[selected_indices_3]
        sorted_data_3= selected_data_3.sort_index()         

        data_final = pd.concat([sorted_data_1, sorted_data_2, sorted_data_3],ignore_index = True)

        # 保存數據到X，並將標籤保存到Y
        numpy_data = data_final.to_numpy()
        numpy_label_data = np.int64(data_label)
        X.append(numpy_data)
        Y.append(numpy_label_data)
        ndarray_data = np.array(X)
        ndarray_label_data = np.array(Y)

# 假設 load_csv_from_csvpath 函數的實現
def load_csv_from_csvpath(data):
    X = []
    y = []
    for _, row in data.iterrows():
        label = row['label']
        csv_path = os.path.join(base_dir, row['csv_paths'])
        csv_data = pd.read_csv(csv_path)  # 加載對應的csv數據
        
        # 將特徵（csv數據）加入X，標籤加入y
        X.append(csv_data)
        y.append(label)
    return y, X



train_data , test_data =train_test_split(df,random_state=123,train_size=0.8)
train_X = load_csv_from_csvpath(train_data)[1]
test_X  = load_csv_from_csvpath(test_data)[1]
train_y = load_csv_from_csvpath(train_data)[0]
test_y =  load_csv_from_csvpath(test_data)[0]

print("訓練集特徵數量:", len(train_X))
print("訓練集標籤數量:", len(train_y))
print("測試集特徵數量:", len(test_X))
print("測試集標籤數量:", len(test_y))
for i, sample in enumerate(train_X):
    print(f"Shape of sample {i}: {sample.shape}")

def create_model():
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape = (30,99)))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(128,activation ='tanh'))
    model.add(Dense(13,activation ='softmax'))

    adam = optimizers.Adam(learning_rate=1e-3)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam')

    return model


# 将标签进行独热编码
train_y_encoded = to_categorical(train_y, num_classes=6)  # 这里的 6 应该是你的类别数
test_y_encoded = to_categorical(test_y, num_classes=6)

model = create_model()
# 假设 train_X 是一个三维数组
# 你可能需要调整 train_X 的形状以匹配 (样本数, 时间步, 特征数)
train_X_reshaped = np.array(train_X).reshape(-1, 45, 99)  # 这里假设你的时间步是 30，特征数是 99

# 开始训练模型
history = model.fit(train_X_reshaped, train_y_encoded, 
                    validation_split=0.2,  # 使用 20% 的训练数据作为验证集
                    epochs=50,  # 训练轮数
                    batch_size=32)  # 批次大小

# 将 test_X 也调整为相同的三维格式
test_X_reshaped = np.array(test_X).reshape(-1, 30, 99)  # 根据你的数据形状调整

# 评估模型
loss, accuracy = model.evaluate(test_X_reshaped, test_y_encoded)
print(f"测试集损失: {loss:.4f}, 测试集准确率: {accuracy:.4f}")

model.save('my_lstm_model.h5')