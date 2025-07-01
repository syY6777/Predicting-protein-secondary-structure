import torch
import torch.nn as nn
import numpy as np
import pandas as pd


def fit(steps, model, loss_f, opt, train_d, test_d,device):
    loss_dic = []
    Vali_loss_dic = []
    for step in range(steps):
        train_loss = 0
        test_loss = 0
        model.train()
        for x, y in train_d:
            x = x.unsqueeze(1).to(device)
            y = y.to(device)
            loss = loss_f(model(x), y)
            train_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_loss /= len(train_d)
        print('Current iteration count' + str(step), 'train loss:' + str(train_loss))
        loss_dic.append(train_loss)

        model.eval()
        pre_dic = []
        label_dic = []
        with torch.no_grad():
            for x, y in test_d:
                x = x.unsqueeze(1).to(device)
                y = y.to(device)
                pre_data = model(x)
                loss2 = loss_f(pre_data, y)
                test_loss += loss2.item()
                pre_data = pre_data.flatten().cpu().numpy()
                y = y.cpu().flatten().numpy()
                # 收集所有信息
                pre_data_reshaped = np.reshape(pre_data, (-1, 3))
                y_reshaped = np.reshape(y, (-1, 3))
                pre_dic.extend(pre_data_reshaped)
                label_dic.extend(y_reshaped)
            test_loss /= len(test_d)
            print('verif loss:' + str(test_loss))
            Vali_loss_dic.append(test_loss)

    torch.save(model.state_dict(), 'Pre trained weight path')
    return loss_dic, pre_dic, label_dic, Vali_loss_dic










