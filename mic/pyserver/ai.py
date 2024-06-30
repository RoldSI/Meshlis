
import pyserver
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class SoundSourceModel(nn.Module):
    def __init__(self):
        super(SoundSourceModel, self).__init__()
        self.fc1 = nn.Linear(3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# For inference later, load the model and run predictions
model = SoundSourceModel().to(device)
model.load_state_dict(torch.load('../../meshlis_ai/sound_source_model.pth'))
model.eval()

s = pyserver.Server()
def process(n):
    if len(n) < 4: return { "x": 0.0, "y": 0.0 }
    # Example inference
    with torch.no_grad():
        print(np.shape(n[2]))
        s = 20.0
        a = np.mean(n[0]) * s; b = np.mean(n[1]) * s; c = np.mean(n[2]) * s;
        new_data = torch.tensor([[a, b, c]], dtype=torch.float32).to(device)
        prediction = model(new_data)[0]
        x = prediction[0]
        y = prediction[1]
        # print(x, y)
        return { "x": x, "y": y }


s.process(process)
s.open_window()


