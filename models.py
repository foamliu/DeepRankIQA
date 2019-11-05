from torch import nn
from torchsummary import summary
from torchvision import models

from config import device


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class DeepIQAModel(nn.Module):
    def __init__(self):
        super(DeepIQAModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear layer
        modules = list(resnet.children())[:-1]
        self.model = nn.Sequential(*modules,
                                   Flatten(),
                                   nn.Dropout(0.5),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(2048, 1),
                                   nn.Sigmoid(),
                                   )

    def forward(self, images):
        x = self.model(images)  # [N, 2048, 1, 1]
        return x


if __name__ == "__main__":
    model = DeepIQAModel().to(device)
    summary(model, input_size=(3, 224, 224))
