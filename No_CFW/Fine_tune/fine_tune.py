import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from data_pre import get_data
from model_stru import CNN
import numpy as np


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


# Fixed random number
torch.manual_seed(24)
np.random.seed(24)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# Load pre_training weights
pretrained_weights = 'Pre_training weight path'
model.load_state_dict(torch.load(pretrained_weights, weights_only=True))

# Freeze all layer parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreezing high-level convolutional layers and fully connected layers
for name, param in model.named_parameters():
    if "conv3" in name or"conv4" in name or "l1" in name :
        param.requires_grad = True

# Set different learning rates
params = [
    {"params": [param for name, param in model.named_parameters() if "conv" not in name], "lr": 0.001},
    {"params": [param for name, param in model.named_parameters() if "conv" in name], "lr": 0.0001}
]
optimizer = optim.Adam(params)
loss_fn = CustomLoss().to(device)



data = pd.read_csv('Fine_tuning test set path', header=None)
train_data, test_data = get_data(data)


# 微调模型
def fine_tune(model, train_dataloader, test_dataloader, loss_fn, optimizer, steps, device):
    loss_dic = []
    test_loss_dic = []
    for step in range(steps):
        train_loss = 0
        test_loss = 0
        model.train()
        for x, y in train_dataloader:
            x = x.unsqueeze(1).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataloader)
        print('Current iteration count' + str(step), 'train loss:' + str(train_loss))
        loss_dic.append(train_loss)

        model.eval()
        pre_dic = []
        label_dic = []
        with torch.no_grad():
            for x, y in test_dataloader:
                x = x.unsqueeze(1).to(device)
                y = y.to(device)
                pre_data  =model(x)
                loss2 = loss_fn(pre_data, y)
                test_loss += loss2.item()
                pre_data =  pre_data .flatten().cpu().numpy()
                y = y.cpu().flatten().numpy()
                pre_data_reshaped = np.reshape(pre_data, (-1, 3))
                y_reshaped = np.reshape(y, (-1, 3))
                pre_dic.extend(pre_data_reshaped)
                label_dic.extend(y_reshaped)
            test_loss /= len(test_dataloader)
            print('verif loss:' + str(test_loss))
            test_loss_dic.append(test_loss)
    return model, pre_dic, label_dic, loss_dic, test_loss_dic


# 微调模型的过程
fine_tuned_model,pre,label,loss,test_loss = fine_tune(model, train_data, test_data, loss_fn,
                                                      optimizer, steps=150, device=device)

# 保存微调后的模型
torch.save(fine_tuned_model.state_dict(), 'Fin_tuning weight path')


loss_dic = pd.DataFrame(loss)
pre_dic = pd.DataFrame(pre)
label_dic = pd.DataFrame(label)
val_loss_dic = pd.DataFrame(test_loss )

loss_dic.to_csv('Training loss file path', index=False)
pre_dic.to_csv('Prediction file path', index=False)
label_dic.to_csv('label file path', index=False)
val_loss_dic.to_csv('Verify loss file path', index=False)
