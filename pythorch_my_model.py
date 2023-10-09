import torch.nn as nn
from torch.optim import Adamax
from torchvision.models import resnet34
from torchsummaryX import summary
import torch

# [Tr] resnet34 modelini tanımlayan sınıf.
# [En] Class that defines the resnet34 model.

class resnet(nn.Module):
    def __init__(self, output):
        super().__init__()
        # [Tr] resnet34 modelini yükler.
        # [En] Loads the resnet34 model.
        self.model = resnet34(pretrained=False)
        # [Tr] resnet34 modelinin çıktı katmanını değiştirir.
        # [En] Changes the output layer of the resnet34 model.
        self.model.fc = torch.nn.Linear(in_features=512, out_features=output)

    # [Tr] modelin ileri beslemesi (forward) işlemi.
    # [En] Forward process of the model.
    def forward(self, x):
        output = self.model(x)
        return output


# [Tr] Cihaz seçimi (GPU).
# [En] Selecting device (GPU).
device = "cuda"

# [Tr] Sınıf sayısı.
# [En] Number of classes.
classes = 12

# [Tr] ResNet modeli oluşturma ve cihaza gönderme.
# [En] Creating and sending ResNet model to device.
model = resnet(classes).to(device=device)

# [Tr] Loss fonksiyonu (CrossEntropyLoss).
# [En] Loss function (CrossEntropyLoss).
criterion = nn.CrossEntropyLoss().to(device=device)
# [Tr] Optimizer (Adamax).
# [En] Adamax optimizer.
optimizer = Adamax(model.parameters(), lr=0.001)

# [Tr] Modelin katmanlarının gösterimi
# [En] Representation of the layers of the model
summary(model, torch.rand((1, 3, 128, 128)).float().to(device))
