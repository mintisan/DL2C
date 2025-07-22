import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

class MNISTNet(nn.Module):
    """简单的CNN模型用于MNIST分类"""
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def train_model():
    """训练MNIST模型"""
    print("开始训练MNIST模型...")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据
    print("加载MNIST数据集...")
    train_dataset = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='../data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 初始化模型 - 优先使用GPU加速
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"使用设备: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备: {device} (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print(f"使用设备: {device} (CPU)")
    
    print(f"🚀 GPU加速训练，可以显著提升训练速度！")
    
    model = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    
    # 训练循环
    print("开始训练...")
    model.train()
    for epoch in range(5):  # 快速训练5个epoch用于演示
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/5, Batch {batch_idx}, Loss: {loss.item():.6f}')
                import sys
                sys.stdout.flush()  # 确保实时输出
        
        print(f'Epoch {epoch+1}/5 完成，平均Loss: {epoch_loss/len(train_loader):.6f}')
        import sys
        sys.stdout.flush()  # 确保实时输出
    
    # 测试模型
    print("测试模型性能...")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'测试准确率: {accuracy:.2f}%')
    print(f'测试损失: {test_loss/len(test_loader):.6f}')
    
    # 保存模型
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/mnist_model.pth')
    torch.save(model, '../models/mnist_model_full.pth')
    print("模型已保存到 ../models/mnist_model.pth")
    
    return model

if __name__ == "__main__":
    trained_model = train_model()
    print("训练完成！") 