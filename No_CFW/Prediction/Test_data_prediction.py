from torch import nn
import torch
from model_stru import CNN
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pickle


def getdata(data):
    # 导入数据
    data = data.values
    test_data = torch.from_numpy(data[:, 3:]).float()
    label = torch.from_numpy(data[:, 0:3]).float()

    # 导入训练集归一化参数
    with open('Fine_tuning parameter/CFW/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # 导入训练集归一化参数
    test_data_norm = torch.tensor(scaler.transform(test_data), dtype=torch.float32)

    # 创建数据加载器
    test_dataset = TensorDataset(test_data_norm, label)
    test_loader = DataLoader(test_dataset,batch_size=5, shuffle=False)

    return test_loader


# 读取测试集
test_datas = pd.read_csv('Test_set/figure3d_data.csv', header=None)
test_datas = getdata(test_datas)

# 实例化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load('Fine_tuning parameter/CFW/fine_tuned_weights.pth', weights_only=True))
# model.load_state_dict(torch.load('Fin_turn/fine_tuned_model_weights.pth'))


model = model.eval()
pre_dic = []
label_dic = []
error_dic = []
with torch.no_grad():
    for x, y in test_datas:
        x = x.unsqueeze(1).to(device)
        pre_data = model(x).flatten().cpu().numpy()

        # 收集所有信息
        pre_data_reshaped = np.reshape(pre_data, (-1, 3))
        pre_data_sum = np.sum(pre_data_reshaped, axis=1, keepdims=True)
        epsilon = 1e-8  # 防止除以零
        pre_data_normalized = 100 * pre_data_reshaped / (pre_data_sum + epsilon)
        pre_dic.extend(pre_data_normalized)


    # 导出信息
    pre = pd.DataFrame(pre_dic)
    pre.to_excel('prediction_figure3d.xlsx', index=False)







