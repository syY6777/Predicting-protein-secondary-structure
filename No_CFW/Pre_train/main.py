import torch
from data_pre import get_data
from model_stru import CNN
from model_train import fit
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np

torch.manual_seed(24)
np.random.seed(24)


class CustomLoss(nn.Module):
    def __init__(self, weight=0.01, threshold=1):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.weight = weight
        self.threshold = threshold  # Determine whether the predicted value is close to zero

    def forward(self, y_pred, y_true):
        # MSE loss
        mse = self.mse_loss(y_pred, y_true)
        # Check if all predicted values are close to zero
        is_near_zero = torch.all(torch.abs(y_pred) < self.threshold, dim=1)

        # Regularization term, not considered when the predicted value approaches zero
        regularization = (torch.sum(y_pred, dim=1) - 100) ** 2
        regularizations = torch.where(is_near_zero, torch.zeros_like(regularization), regularization)
        total_loss = mse + self.weight * torch.mean(regularizations)
        return total_loss



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)



data = pd.read_csv('Your dataset', header=None)
train_data, val_data = get_data(data)



loss_fn = CustomLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.01)

loss, pre, label,val_loss = fit(steps=200, model=model, loss_f=loss_fn, opt=optimizer, train_d=train_data,
                                 test_d=val_data, device=device)

loss_dic = pd.DataFrame(loss)
pre_dic = pd.DataFrame(pre)
label_dic = pd.DataFrame(label)
val_dic = pd.DataFrame(val_loss)

loss_dic.to_excel('Training loss file path', index=False)
pre_dic.to_excel('Prediction file path', index=False)
label_dic.to_excel('label file path', index=False)
val_dic.to_excel('Verify loss file path', index=False)
