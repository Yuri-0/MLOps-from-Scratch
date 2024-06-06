## Quick Start

### Server Preparation
This model checkpoint management system support to run as Docker and source build. The staging_server directory contains files to run API from docker or source, and the client directory contains file for ML model train, inference and test API using the SDK. Please place each directory to suitable path before start.

#### Docker
```
cd staging_server
docker build . -t model_manager
docker run --net=host model_manager
```
Build docker image using `dockerfile` and run it. In default setting, the docker container listen model log or load request from port `8000`.

#### Source
```
cd staging_server
pip install -r requirements.txt
uvicorn api_server:app --reload
```
Install requirements using `pip` and start API with `uvicorn`.

### Tutorial
To run the tutorial, you need to install all dependency packages and complete the server preparation steps to enable communication with the system server. In this tutorial, we will implement how to train an image classification model using MNIST dataset and pytorch lightning, register it in the system, and load the registered model for inference.

### Training the Model

The code is located in `client/mnist_train_lightning.py` and explained as follows.

```
from model_manager_sdk import ModelManager, TotalLogger

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

m_manager = ModelManager(endpoint='http://localhost:8000', user_id='your_id', model_name='your_model')
```

- The plugins required to register data such as model training information and model weights with the system are imported from the `model_manager_sdk`.
- `ModelManager` is a required module to use sub-functions such as model weights and hyperparameter registration. By default, it is defined with the endpoint of the system server, model user ID and model name, etc. as arguments.
- `TotalLogger` is a feature similar to MLflow's `AutoLogger`, which registers as much information as possible in the system, including information about the model being trained and the operating environment.
- Individual models registered in the system are identified based on `user_id`, `model_name`, and `version`, which are given as arguments to `ModelManager` or sub-functions.

> ℹ️ When the same user registers two different models with the system, they must use different model names. If the same name is used, the versions are distinguished by the order in which the training was run.

```
PATH_DATASETS = '.'
AVAIL_GPUS = torch.cuda.device_count()
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class LitMNIST(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean()

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)
```

- Define the model structure and training method, and how data is loaded and processed.
- If `TotalLogger` does not support your purpose, or if you want to use an individual Logger to register model and training information, you can modify your code to include Logger calls during the training progression.

```
model = LitMNIST()
trainer = Trainer(
    max_epochs=3,
    callbacks=[TotalLogger(logger=m_manager)]
)
trainer.fit(model)
```

- Define a `Trainer` with training arguments and `fit` the model LitMNIST.
- `TotalLogger` is used as a custom callback for the Trainer.
    - All Loggers provided by the SDK send the information they collect to the system server in real time during the training process.

### Model Serving

The code is located in `client/mnist_inference_lightning.py` and explained as follows.

```
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
```

- Define the structure of the model to be served. Since the model artifacts stored in the system are weights data, defining the model structure in advance is necessary for normal model loading.
    - Logger allows you to store training code along with model artifacts, so you can refer to the stored code to write them when needed.
- Preprocess the incoming data for inference into a form that the trained model can handle.

```
model = LitMNIST()

checkpoint = m_manager.load_models('your_model')
model.load_state_dict(checkpoint)

model.eval()
with torch.no_grad():
    prediction = model(pick_data)
    prediction = torch.argmax(prediction, dim=1)

print(f'Index of data: {rand_index}\nPrediction: {prediction[0]} / Label: {int(pick_answer[0])}')
```

- `load_models` takes a `model_name` and `version` (latest version if undefined) and `ModelManager`'s `user_name` to search unique artifacts and load their weights.
- It reflects and infers the loaded weights in a predefined model structure.
