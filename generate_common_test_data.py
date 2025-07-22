#!/usr/bin/env python3
"""
生成三种语言通用的测试数据
确保Python、C++、C使用完全相同的输入进行对比
"""

import numpy as np
import torchvision
import torchvision.transforms as transforms
import json
import os
from pathlib import Path

def generate_common_test_data(num_samples=10):
    """生成通用测试数据"""
    print("🔄 生成通用测试数据...")
    
    # 创建测试数据目录
    test_data_dir = Path("./test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # 加载MNIST测试数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    try:
        test_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        print("✅ MNIST数据集加载成功")
    except Exception as e:
        print(f"❌ MNIST数据集加载失败: {e}")
        return None
    
    # 准备测试数据
    test_samples = []
    
    # 选择固定的样本索引，确保结果可重现
    np.random.seed(42)  # 固定随机种子
    sample_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        image, label = test_dataset[idx]
        
        # 转换为numpy数组
        image_np = image.squeeze().numpy()  # 移除batch维度，得到[28, 28]
        
        # 保存原始数据（0-1范围）
        sample_data = {
            'sample_id': i,
            'true_label': int(label),
            'image_shape': [28, 28],
            'pixel_values': image_np.flatten().tolist(),  # 展平为一维数组便于存储
            'mnist_index': int(idx)
        }
        
        test_samples.append(sample_data)
        
        # 同时保存为二进制文件供C/C++使用
        binary_file = test_data_dir / f"sample_{i:02d}.bin"
        image_np.astype(np.float32).tofile(binary_file)
        
        print(f"样本 {i}: 标签={label}, MNIST索引={idx}, 保存到={binary_file}")
    
    # 保存元数据
    metadata = {
        'num_samples': num_samples,
        'image_shape': [28, 28],
        'data_type': 'float32',
        'pixel_range': [0.0, 1.0],
        'description': 'MNIST测试样本，用于三种语言推理对比',
        'samples': test_samples
    }
    
    metadata_file = test_data_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 生成了 {num_samples} 个测试样本")
    print(f"📁 数据保存在: {test_data_dir}")
    print(f"📄 元数据文件: {metadata_file}")
    
    return metadata

def verify_test_data():
    """验证生成的测试数据"""
    print("\n🔍 验证测试数据...")
    
    test_data_dir = Path("./test_data")
    metadata_file = test_data_dir / "metadata.json"
    
    if not metadata_file.exists():
        print("❌ 元数据文件不存在")
        return False
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"测试样本数量: {metadata['num_samples']}")
    print(f"图像尺寸: {metadata['image_shape']}")
    print(f"数据类型: {metadata['data_type']}")
    
    # 验证二进制文件
    for i in range(metadata['num_samples']):
        binary_file = test_data_dir / f"sample_{i:02d}.bin"
        if binary_file.exists():
            # 读取并验证
            image_data = np.fromfile(binary_file, dtype=np.float32)
            expected_size = metadata['image_shape'][0] * metadata['image_shape'][1]
            
            if len(image_data) == expected_size:
                print(f"✅ 样本 {i}: {binary_file.name} 验证通过")
            else:
                print(f"❌ 样本 {i}: 大小不匹配，期望={expected_size}, 实际={len(image_data)}")
        else:
            print(f"❌ 样本 {i}: {binary_file} 不存在")
    
    return True

if __name__ == "__main__":
    print("🎯 生成三种语言通用测试数据")
    print("=" * 50)
    
    # 生成测试数据
    metadata = generate_common_test_data(num_samples=10)
    
    if metadata:
        # 验证测试数据
        verify_test_data()
        
        print("\n🎉 测试数据生成完成！")
        print("现在可以修改三种语言的推理代码来使用相同的数据")
    else:
        print("❌ 测试数据生成失败") 