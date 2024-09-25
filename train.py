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


# 讀取 CSV 文件到 DataFrame
csv_file_path = 'D:\\0925\\label_csv.csv'  # 替換為你的 CSV 文件路徑
df = pd.read_csv(csv_file_path)

# 確認讀取的數據結構
print(df.head())


def load_csv_from_csvpath(df):
    num_samples = 15
    frame_length = 50

    X = []
    Y = []

    df = df.dropna(subset=['label'])

    for index, item_label, item_csvpath in df.values:
        try:
            csv_data = pd.read_csv(item_csvpath)  # 加载每个 CSV 文件
        except FileNotFoundError:
            print(f"文件 {item_csvpath} 不存在，跳过该行")
            continue  # 如果文件不存在，跳过该行
        
        if len(csv_data) < frame_length:
            print(f"文件 {item_csvpath} 的数据不足 {frame_length} 帧，跳过该行")
            continue  # 如果帧数不足，跳过该行
            
        data = csv_data  # 直接使用读取的 csv_data

        # 从数据中抽样
        data_1 = data.iloc[:int(len(data) * 0.2)]
        selected_indices_1 = np.random.choice(data_1.index, size=num_samples, replace=False)
        selected_data_1 = data_1.loc[selected_indices_1]

        data_2 = data.iloc[int(len(data) * 0.2):int(len(data) * 0.4)]
        selected_indices_2 = np.random.choice(data_2.index, size=num_samples, replace=False)
        selected_data_2 = data_2.loc[selected_indices_2]

        data_3 = data.iloc[-int(len(data) * 0.4):]
        selected_indices_3 = np.random.choice(data_3.index, size=num_samples, replace=False)
        selected_data_3 = data_3.loc[selected_indices_3]

        data_final = pd.concat([selected_data_1, selected_data_2, selected_data_3], ignore_index=True)

        # 保存数据到X，并将标签保存到Y
        numpy_data = data_final.to_numpy()
        numpy_label_data = np.int64(item_label)  # 直接使用 item_label
        X.append(numpy_data)
        Y.append(numpy_label_data)

    # 转换为 ndarray
    ndarray_data = np.array(X)
    ndarray_label_data = np.array(Y)

    print(ndarray_data.shape)
    print(ndarray_label_data.shape)
    return ndarray_label_data, ndarray_data  # 返回标签和特征    

        
# 假設基礎目錄是 D:\\0925
base_dir = 'D:\\0925'
'''
# 假設 load_csv_from_csvpath 函數的實現
def load_csv_from_csvpath(data):
    X = []
    y = []
    for _, row in data.iterrows():
        label = row['label']
        csv_path = os.path.join(base_dir, row['csv_paths'])
        csv_data = pd.read_csv(csv_path)  # 加載對應的csv數據
        if not os.path.exists(csv_path):
            print(f"文件 {csv_path} 不存在，跳過該行")
            continue
        csv_data = pd.read_csv(csv_path)  # 加載對應的csv數據
        
        # 將特徵（csv數據）加入X，標籤加入y
        X.append(csv_data)
        y.append(label)
    return y, X

'''

train_data , test_data =train_test_split(df,random_state=123,train_size=0.8)
train_X = load_csv_from_csvpath(train_data)[1]
test_X  = load_csv_from_csvpath(test_data)[1]
train_y = load_csv_from_csvpath(train_data)[0]
test_y =  load_csv_from_csvpath(test_data)[0]

# 假设你的时间步是 45，特征数是 99
# 你需要确保 train_X_reshaped 和 test_X_reshaped 都是三维数组
train_X_reshaped = np.concatenate(train_X).reshape(-1, 45, 99)
test_X_reshaped = np.concatenate(test_X).reshape(-1, 45, 99)

print("訓練集特徵數量:", len(train_X))
print("訓練集標籤數量:", len(train_y))
print("測試集特徵數量:", len(test_X))
print("測試集標籤數量:", len(test_y))

for i, sample in enumerate(train_X):
    print(f"Shape of sample {i}: {sample.shape}")


def create_model():
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape = (45,99)))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(128,activation ='tanh'))
    model.add(Dense(6,activation ='softmax'))

    adam = optimizers.Adam(learning_rate=1e-3)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

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
test_X_reshaped = np.array(test_X).reshape(-1, 45, 99)  # 根据你的数据形状调整


print(f"测试数据形状: {test_X_reshaped.shape}")
print(f"测试标签形状: {test_y_encoded.shape}")

# 评估模型
loss, accuracy = model.evaluate(test_X_reshaped, test_y_encoded)
print(f"测试集损失: {loss:.4f}, 测试集准确率: {accuracy:.4f}")

model.save('my_lstm_model.h5')