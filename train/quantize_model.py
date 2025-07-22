import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import os
import numpy as np
from train_model import MNISTNet

def simulate_quantization(model, bits=8):
    """
    模拟量化：通过限制权重精度来模拟量化效果
    这是一个教学用的简化量化实现
    """
    print(f"模拟 {bits}-bit 量化...")
    
    # 创建模型副本
    quantized_model = MNISTNet()
    quantized_model.load_state_dict(model.state_dict())
    quantized_model.eval()
    
    # 对每个参数进行"量化"
    with torch.no_grad():
        for name, param in quantized_model.named_parameters():
            if 'weight' in name:
                # 计算量化范围
                param_min = param.min()
                param_max = param.max()
                
                # 量化到指定位数
                levels = 2 ** bits
                scale = (param_max - param_min) / (levels - 1)
                
                # 量化和反量化
                quantized_param = torch.round((param - param_min) / scale) * scale + param_min
                param.copy_(quantized_param)
                
                print(f"量化参数 {name}: 范围 [{param_min:.4f}, {param_max:.4f}], 缩放 {scale:.6f}")
    
    return quantized_model

def get_model_size(model):
    """计算模型大小"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def test_model_accuracy(model, data_loader):
    """测试模型精度"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total

def quantize_model():
    """执行模拟量化"""
    print("开始模型量化（兼容性版本）...")
    
    # 加载原始模型
    print("加载原始模型...")
    model = MNISTNet()
    model.load_state_dict(torch.load('../models/mnist_model.pth', map_location='cpu', weights_only=True))
    model.eval()
    
    # 准备测试数据
    print("准备测试数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='../data', train=False, download=False, transform=transform
    )
    
    # 使用少量数据进行测试
    test_subset = torch.utils.data.Subset(test_dataset, range(1000))
    test_loader = DataLoader(test_subset, batch_size=100, shuffle=False)
    
    # 测试原始模型
    print("测试原始模型精度...")
    original_accuracy = test_model_accuracy(model, test_loader)
    
    # 执行模拟量化
    print("执行模拟量化...")
    quantized_model = simulate_quantization(model, bits=8)
    
    # 测试量化模型
    print("测试量化模型精度...")
    quantized_accuracy = test_model_accuracy(quantized_model, test_loader)
    
    # 计算模型大小
    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)
    
    # 显示结果
    print(f"\n=== 模拟量化结果对比 ===")
    print(f"量化方法: 8-bit 模拟量化 (兼容性版本)")
    print(f"原始模型精度: {original_accuracy:.2f}%")
    print(f"量化模型精度: {quantized_accuracy:.2f}%")
    print(f"精度损失: {original_accuracy - quantized_accuracy:.2f}%")
    print(f"原始模型大小: {original_size:.2f} MB")
    print(f"量化模型大小: {quantized_size:.2f} MB")
    print(f"模型压缩比: {original_size/quantized_size:.2f}x")
    
    print(f"\n💡 注意: 这是教学用的模拟量化，实际部署建议使用支持的平台进行真实量化")
    print(f"📚 在生产环境中，可以使用 ONNX Runtime 的量化功能")
    
    # 保存量化模型
    print("保存模拟量化模型...")
    os.makedirs('../models', exist_ok=True)
    torch.save(quantized_model.state_dict(), '../models/mnist_quantized.pth')
    torch.save(quantized_model, '../models/mnist_quantized_full.pth')
    print("量化模型已保存到 ../models/mnist_quantized.pth")
    
    return model, quantized_model

if __name__ == "__main__":
    try:
        original, quantized = quantize_model()
        print("✅ 模拟量化完成！")
    except Exception as e:
        print(f"❌ 量化过程出错: {e}")
        print("\n🔧 可能的解决方案:")
        print("1. 确保模型文件存在: ../models/mnist_model.pth")
        print("2. 确保数据文件存在: ../data/MNIST/")
        import sys
        sys.exit(1) 