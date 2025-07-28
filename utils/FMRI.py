
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, model_type: str = "c", num_classes: int = 4):
        super().__init__()
        if model_type == "f":
            self.conv1, self.conv2, self.conv3 = nn.Conv2d(3,16,3,1,1), nn.Conv2d(16,32,3,1,1), nn.Conv2d(32,64,3,1,1)
            self.fc1 = nn.Linear(64 * 28 * 28, 256)
        elif model_type == "c":
            self.conv1, self.conv2, self.conv3 = nn.Conv2d(3,32,3,1,1), nn.Conv2d(32,64,3,1,1), nn.Conv2d(64,128,3,1,1)
            self.fc1 = nn.Linear(128 * 28 * 28, 512)
        elif model_type == "q":
            self.conv1, self.conv2, self.conv3 = nn.Conv2d(3,64,3,1,1), nn.Conv2d(64,128,3,1,1), nn.Conv2d(128,256,3,1,1)
            self.conv4 = nn.Conv2d(256,512,3,1,1)
            self.fc1 = nn.Linear(512 * 14 * 14, 1024)
        else:
            raise ValueError("model_type must be f / c / q")

        self.dropout, self.relu, self.pool = nn.Dropout(0.5), nn.ReLU(), nn.MaxPool2d(2,2)
        self.fc2 = nn.Linear(self.fc1.out_features, num_classes) 

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        if hasattr(self, "conv4"):
            x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)