from torch import nn
from torchsummary import summary
from torchvision import models

from config import device, num_classes


class DeepIQAModel(nn.Module):
    def __init__(self):
        super(DeepIQAModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 1, 1]
        x = x.view(-1, 2048)  # [N, 2048]
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = DeepIQAModel().to(device)
    summary(model, input_size=(3, 224, 224))
