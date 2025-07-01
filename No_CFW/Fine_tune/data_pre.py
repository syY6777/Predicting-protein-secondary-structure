import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


def get_data(data, export_test=True):
    original_indices = data.index.to_numpy()
    features = data.iloc[:, 3:].to_numpy()
    labels = data.iloc[:, 0:3].to_numpy()

    x_train_val, x_test, y_train_val, y_test, train_val_indices, test_indices = train_test_split(
        features, labels, original_indices, test_size=0.2, random_state=24)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=2/8, random_state=24)

    if export_test:
        test_data_df = pd.DataFrame(x_test, columns=[f"Feature_{i}" for i in range(x_test.shape[1])])
        test_data_df['Label1'] = y_test[:, 0]
        test_data_df['Label2'] = y_test[:, 1]
        test_data_df['Label3'] = y_test[:, 2]
        test_data_df['Original Index'] = test_indices
        test_data_df.to_csv('Test set save path', index=False)
        print("The test set has been saved")

    # normalization
    scaler = StandardScaler()
    x_train_normalized = scaler.fit_transform(x_train)
    x_val_normalized = scaler.transform(x_val)


    # Save the scaler object
    with open('Scaler saves path', 'wb') as f:
        pickle.dump(scaler, f)

    train_dataset = TensorDataset(torch.tensor(x_train_normalized, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(x_val_normalized, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=5, shuffle=False)


    return train_dataloader, val_dataloader
