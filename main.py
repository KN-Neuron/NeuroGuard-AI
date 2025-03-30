import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from eeg_lib.data.data_loader.custom_data_loader import get_raw_rest_data

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=32, num_classes=10, conv_features=64):
        super(LSTMModel, self).__init__()

        self.conv = nn.Conv1d(input_size, conv_features, kernel_size=3, padding="same")
        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(conv_features, 128, batch_first=True)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):        
        x = self.conv(x)
        x = self.pool(x)

        x = x.permute(0, 2, 1)

        _, (h_n, _) = self.lstm(x)

        x = self.fc1(h_n[-1])
        x = F.relu(x)

        x: torch.Tensor = self.fc2(x)
        # x = self.softmax(x)

        return x.softmax(dim=1)


class CustomDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X, self.y = get_raw_rest_data(pre_path="../../artificial-intelligence")
        
        self.X = list(map(lambda x: torch.tensor(x, dtype=torch.float32), self.X))
        
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = torch.tensor(OneHotEncoder(sparse_output=False).fit_transform(self.y.reshape((-1, 1))))
        
        
        self.num_classes = len(self.y[0])
        self.input_size = self.X[0].shape[0]
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
def collate_fn(batch):
    X_batch, y_batch = zip(*batch)
    return list(X_batch), torch.stack(y_batch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CustomDataset()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

input_size = dataset.input_size
num_classes = dataset.num_classes

model = LSTMModel(input_size=input_size, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    model.train()

    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        batch_y = batch_y.to(device)

        outputs = []
        for x in batch_X:
            x = x.to(device)
            output = model(x.unsqueeze(0)) 
            outputs.append(output)

        outputs = torch.cat(outputs, dim=0) 
        
        loss = criterion(outputs, batch_y.to(device))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_y = batch_y.to(device)

            outputs = []
            for x in batch_X:
                x = x.to(device)
                output = model(x.unsqueeze(0))
                outputs.append(output)

            outputs = torch.cat(outputs, dim=0)

            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(batch_y, 1)

            correct += (predicted == labels).sum().item()
            total += batch_y.size(0)

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1:2}/50], Loss: {epoch_loss:.4f}, Accuracy: {correct / total:.2f}%")
