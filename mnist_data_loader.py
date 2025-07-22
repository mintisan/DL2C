#!/usr/bin/env python3
"""
MNIST原始数据加载器
直接读取MNIST原始格式数据，为三种语言提供统一的测试数据
"""

import struct
import numpy as np
import json
import os
from pathlib import Path

class MNISTDataLoader:
    """MNIST数据加载器"""
    
    def __init__(self, data_dir="./data/MNIST/raw"):
        self.data_dir = Path(data_dir)
        
    def load_images(self, filename):
        """加载MNIST图像文件"""
        filepath = self.data_dir / filename
        
        with open(filepath, 'rb') as f:
            # 读取文件头
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            
            if magic != 2051:
                raise ValueError(f"Invalid magic number: {magic}")
            
            print(f"加载图像: {num_images} 张，尺寸: {rows}x{cols}")
            
            # 读取图像数据
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)
            
            # 归一化到 [0, 1]
            images = images.astype(np.float32) / 255.0
            
            return images
    
    def load_labels(self, filename):
        """加载MNIST标签文件"""
        filepath = self.data_dir / filename
        
        with open(filepath, 'rb') as f:
            # 读取文件头
            magic, num_labels = struct.unpack('>II', f.read(8))
            
            if magic != 2049:
                raise ValueError(f"Invalid magic number: {magic}")
            
            print(f"加载标签: {num_labels} 个")
            
            # 读取标签数据
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            
            return labels
    
    def create_test_subset(self, num_samples=100, random_seed=42):
        """创建测试子集，确保结果可重现"""
        print(f"🔄 创建包含 {num_samples} 个样本的测试子集...")
        
        # 加载完整的测试数据
        test_images = self.load_images("t10k-images-idx3-ubyte")
        test_labels = self.load_labels("t10k-labels-idx1-ubyte")
        
        # 使用固定种子选择样本
        np.random.seed(random_seed)
        indices = np.random.choice(len(test_images), num_samples, replace=False)
        indices = np.sort(indices)  # 排序以保证一致性
        
        selected_images = test_images[indices]
        selected_labels = test_labels[indices]
        
        print(f"✅ 选择了 {len(selected_images)} 个样本")
        print(f"标签分布: {np.bincount(selected_labels)}")
        
        return selected_images, selected_labels, indices
    
    def save_for_inference(self, images, labels, indices, output_dir="./test_data_mnist"):
        """保存数据供三种语言推理使用"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"💾 保存数据到: {output_dir}")
        
        # 保存为二进制文件供C/C++使用
        for i, (image, label, orig_idx) in enumerate(zip(images, labels, indices)):
            # 保存图像数据
            binary_file = output_dir / f"image_{i:03d}.bin"
            image.astype(np.float32).tofile(binary_file)
            
            print(f"样本 {i:3d}: 标签={label}, 原始索引={orig_idx}, 文件={binary_file.name}")
        
        # 保存元数据
        metadata = {
            'num_samples': len(images),
            'image_shape': [28, 28],
            'data_type': 'float32',
            'pixel_range': [0.0, 1.0],
            'description': 'MNIST测试子集，用于三种语言推理对比',
            'random_seed': 42,
            'samples': []
        }
        
        for i, (label, orig_idx) in enumerate(zip(labels, indices)):
            metadata['samples'].append({
                'sample_id': i,
                'true_label': int(label),
                'original_mnist_index': int(orig_idx),
                'image_file': f"image_{i:03d}.bin"
            })
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 保存为NPZ文件供Python使用
        npz_file = output_dir / "mnist_test_subset.npz"
        np.savez(npz_file, 
                images=images, 
                labels=labels, 
                indices=indices)
        
        print(f"✅ 保存完成:")
        print(f"  - 二进制文件: {len(images)} 个 *.bin 文件")
        print(f"  - 元数据: {metadata_file}")
        print(f"  - Python数据: {npz_file}")
        
        return metadata
    
    def verify_data_consistency(self, output_dir="./test_data_mnist"):
        """验证保存的数据一致性"""
        print("\n🔍 验证数据一致性...")
        
        output_dir = Path(output_dir)
        
        # 加载元数据
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # 加载NPZ文件
        npz_file = output_dir / "mnist_test_subset.npz"
        npz_data = np.load(npz_file)
        
        print(f"元数据样本数: {metadata['num_samples']}")
        print(f"NPZ数据形状: {npz_data['images'].shape}")
        
        # 验证二进制文件
        success_count = 0
        for i in range(metadata['num_samples']):
            binary_file = output_dir / f"image_{i:03d}.bin"
            if binary_file.exists():
                # 读取二进制数据
                bin_data = np.fromfile(binary_file, dtype=np.float32).reshape(28, 28)
                npz_data_sample = npz_data['images'][i]
                
                # 比较数据
                if np.allclose(bin_data, npz_data_sample):
                    success_count += 1
                else:
                    print(f"❌ 样本 {i} 数据不一致")
        
        print(f"✅ 验证完成: {success_count}/{metadata['num_samples']} 个文件一致")
        
        return success_count == metadata['num_samples']

def main():
    """主函数"""
    print("🎯 MNIST原始数据加载器")
    print("=" * 50)
    
    # 检查数据目录
    data_dir = "./data/MNIST/raw"
    if not os.path.exists(data_dir):
        print(f"❌ MNIST数据目录不存在: {data_dir}")
        print("请确保已下载MNIST数据集")
        return
    
    # 创建数据加载器
    loader = MNISTDataLoader(data_dir)
    
    # 创建测试子集（可以调整样本数量）
    num_samples = 100  # 可以改为更大的数字，比如1000或10000
    images, labels, indices = loader.create_test_subset(num_samples=num_samples)
    
    # 保存数据
    metadata = loader.save_for_inference(images, labels, indices)
    
    # 验证数据
    if loader.verify_data_consistency():
        print("\n🎉 数据准备完成！")
        print("现在可以使用原始MNIST数据进行三种语言的推理对比")
        print(f"测试样本数: {num_samples}")
        print("使用方法:")
        print("  1. Python: 直接加载 mnist_test_subset.npz")
        print("  2. C/C++: 读取 image_*.bin 文件和 metadata.json")
    else:
        print("❌ 数据验证失败")

if __name__ == "__main__":
    main() 