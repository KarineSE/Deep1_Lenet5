import torch
import torchvision
from torch import nn  # building blocks of a CNN
from torchvision.transforms import Resize, ToTensor, Normalize, Compose

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


class Lenet5(nn.Module):
    def __init__(self, reg):
        super(Lenet5, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(1024, 128)
        # self.fc2 = nn.Linear(128, 10)

        if reg == "none" or reg == "wd":
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.fc1 = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU()
            )
            self.fc2 = nn.Sequential(
                nn.Linear(120, 84),
                nn.ReLU()
            )
            self.fc3 = nn.Linear(84, 10)

        if reg.lower() == "nb":  # add batch normalization after conv before activation
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.fc1 = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.BatchNorm1d(num_features=120),
                nn.ReLU()
            )
            self.fc2 = nn.Sequential(
                nn.Linear(120, 84),
                nn.BatchNorm1d(num_features=84),
                nn.ReLU()
            )
            self.fc3 = nn.Linear(84, 10)

        if reg.lower() == "dr":  # add dropout after activation before MaxPooling (one before last layer)
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.fc1 = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.Dropout(0.3),
                nn.ReLU()
            )
            self.fc2 = nn.Sequential(
                nn.Linear(120, 84),
                nn.Dropout(0.3),
                nn.ReLU()
            )
            self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim=1)

        return x


def load_dataset(dataset_part: str, data_dir: str, custom_transforms=None ) -> \
        torch.utils.data.Dataset:
    """Loads dataset part from dataset name.
    Args:
        dataset_name: dataset name
        dataset_part: dataset part, one of: train, val, test.
    Returns:
        dataset: a torch.utils.dataset.Dataset instance.
    """
    if custom_transforms is not None:
        transform = custom_transforms
    else:
        transform = {"train": Compose([ToTensor(), Normalize(mean=0.3814, std=0.3994)]),
                     "val": Compose([ToTensor(), Normalize(mean=0.3814, std=0.3994)]),
                     "test": Compose([ToTensor(), Normalize(mean=0.3814, std=0.3994)])
                     }
    dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True,
                                                transform=transform[dataset_part],
                                                download=True)
    return dataset


def load_model(model_name: str, reg: str) -> nn.Module:
    """Load the model corresponding to the name given.
    Args:
        Reg: regularization technique.
    Returns:
        model: the model initialized, and loaded to device.
    """
    print(f"Building model {model_name}...")
    model = Lenet5(reg)
    model = model.to(device)
    return model
