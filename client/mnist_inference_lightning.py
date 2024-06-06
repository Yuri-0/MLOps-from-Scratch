from model_manager_sdk import ModelManager

import torch
import random
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import MNIST


class LitMNIST(LightningModule):
    def __init__(self, hidden_size=64):

        super().__init__()

        # Hardcode some dataset specific attributes
        num_classes = 10
        dims = (1, 28, 28)
        channels, width, height = dims

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)


m_manager = ModelManager(endpoint='http://localhost:8000', user_id='your_id')

PATH_DATASETS = '.'

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

mnist = MNIST(PATH_DATASETS, train=False, transform=transform)

rand_index = random.randint(0, mnist.data.size()[0] - 1)
pick_data = mnist.data[rand_index:rand_index+1].type(torch.float)
pick_answer = mnist.targets[rand_index:rand_index+1].type(torch.float32)

model = LitMNIST()

checkpoint = m_manager.load_models('your_model')
model.load_state_dict(checkpoint)

model.eval()
with torch.no_grad():
    prediction = model(pick_data)
    prediction = torch.argmax(prediction, dim=1)

print(f'Index of data: {rand_index}\nPrediction: {prediction[0]} / Label: {int(pick_answer[0])}')
