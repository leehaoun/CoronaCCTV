import torch
import torch.nn as nn  # 뉴럴 네트워크를 생성하기 위한 패키지
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(0)  # 랜덤 시드를 준다
device = 'cuda'
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()


        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )


        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )

        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )

        self.layer8 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(8192, 8)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((200, 200))
        ]
    )
    batch_size=32
    train_datasets = datasets.ImageFolder('./trainset', transform=transform)
    train_loader = DataLoader(train_datasets, batch_size, shuffle=False)
    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    losses = []
    epochs = 100
    n = len(train_loader)
    model.train().to(device)  
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            loss.backward()        
            optimizer.step()
            running_loss += loss.item()
            print('epoch:[%d/%d] batch:[%d/%d]' %(epoch + 1,epochs, i+1,len(train_loader)))
        losses.append(running_loss/n)
    PATH = './8192_weights.pth' # 모델 저장 경로
    torch.save(model.state_dict(), PATH) # 모델 저장
    plt.plot(losses)
    plt.title(loss)
    plt.xlabel('epoch')
    plt.show()    
    print('Train Done, Save .pth')
        
    

