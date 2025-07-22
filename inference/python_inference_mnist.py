#!/usr/bin/env python3
"""
Python ONNX推理 - 使用真实MNIST测试数据
使用从原始MNIST测试集中选择的真实样本进行推理
"""

import onnxruntime as ort
import numpy as np
import json
import time
import os
from pathlib import Path

class PythonONNXInferenceMNIST:
    """Python ONNX推理类 - 使用真实MNIST数据"""
    
    def __init__(self, model_path):
        """初始化ONNX推理引擎"""
        print(f"加载ONNX模型: {model_path}")
        
        # 创建ONNX Runtime会话
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"✅ Python ONNX Runtime初始化成功")
        print(f"输入名称: {self.input_name}")
        print(f"输出名称: {self.output_name}")
        
    def preprocess(self, image_data):
        """预处理图像数据"""
        # 输入数据已经是[28, 28]的float32数组，范围[0,1]
        # 需要标准化并调整形状
        
        # 标准化 (MNIST标准化参数)
        mean = 0.1307
        std = 0.3081
        normalized = (image_data - mean) / std
        
        # 调整形状为 [1, 1, 28, 28]
        input_data = normalized.reshape(1, 1, 28, 28).astype(np.float32)
        
        return input_data
    
    def postprocess(self, output):
        """后处理输出结果"""
        # 获取ONNX输出
        logits = output[0][0]  # 从 [1, 10] 变为 [10]
        
        # 应用softmax获得概率分布
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # 获取预测类别和置信度
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return {
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'probabilities': probabilities.tolist(),
            'raw_logits': logits.tolist()
        }
    
    def inference(self, image_data):
        """执行推理"""
        start_time = time.time()
        
        # 预处理
        processed_input = self.preprocess(image_data)
        
        # 运行推理
        ort_inputs = {self.input_name: processed_input}
        ort_outputs = self.session.run([self.output_name], ort_inputs)
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 后处理
        result = self.postprocess(ort_outputs)
        result['inference_time_ms'] = inference_time
        
        return result

def load_mnist_test_data():
    """加载MNIST测试数据"""
    test_data_dir = Path("../test_data_mnist")
    npz_file = test_data_dir / "mnist_test_subset.npz"
    
    if not npz_file.exists():
        print("❌ 找不到MNIST测试数据，请先运行 mnist_data_loader.py")
        return None, None, None
    
    # 加载NPZ数据
    data = np.load(npz_file)
    images = data['images']  # shape: (num_samples, 28, 28)
    labels = data['labels']  # shape: (num_samples,)
    indices = data['indices']  # 原始MNIST索引
    
    print(f"🔍 加载 {len(images)} 个MNIST测试样本")
    print(f"数据形状: {images.shape}")
    print(f"标签分布: {np.bincount(labels)}")
    
    return images, labels, indices

def test_python_inference_mnist():
    """使用真实MNIST数据进行Python推理测试"""
    print("=== Python ONNX推理测试 (真实MNIST数据) ===")
    
    # 加载模型
    model_path = '../models/mnist_model.onnx'
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    inference_engine = PythonONNXInferenceMNIST(model_path)
    
    # 加载MNIST测试数据
    images, labels, indices = load_mnist_test_data()
    if images is None:
        return None
    
    print(f"\n开始推理 {len(images)} 个样本...")
    
    # 执行推理
    results = []
    correct_predictions = 0
    total_time = 0
    
    for i, (image_data, true_label, original_idx) in enumerate(zip(images, labels, indices)):
        # 执行推理
        result = inference_engine.inference(image_data)
        
        # 检查准确性
        is_correct = result['predicted_class'] == true_label
        if is_correct:
            correct_predictions += 1
        
        # 记录结果
        sample_result = {
            'sample_id': i,
            'original_mnist_index': int(original_idx),
            'true_label': int(true_label),
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'inference_time_ms': result['inference_time_ms'],
            'is_correct': bool(is_correct),
            'probabilities': result['probabilities']
        }
        
        results.append(sample_result)
        total_time += result['inference_time_ms']
        
        # 显示进度（每10个样本）
        if (i + 1) % 10 == 0:
            print(f"完成 {i+1:3d}/{len(images)} 样本，当前准确率: {correct_predictions/(i+1)*100:.1f}%")
    
    # 计算统计信息
    accuracy = correct_predictions / len(results)
    avg_time = total_time / len(results)
    std_time = np.std([r['inference_time_ms'] for r in results])
    
    print(f"\n=== 推理结果统计 ===")
    print(f"总样本数: {len(results)}")
    print(f"正确预测: {correct_predictions}")
    print(f"准确率: {accuracy:.2%}")
    print(f"平均推理时间: {avg_time:.2f} ms")
    print(f"时间标准差: {std_time:.2f} ms")
    print(f"推理速度: {1000/avg_time:.1f} FPS")
    
    # 显示错误样本
    wrong_samples = [r for r in results if not r['is_correct']]
    if wrong_samples:
        print(f"\n❌ 错误预测样本 ({len(wrong_samples)} 个):")
        for sample in wrong_samples[:5]:  # 只显示前5个
            print(f"  样本 {sample['sample_id']:3d}: 真实={sample['true_label']}, "
                  f"预测={sample['predicted_class']}, 置信度={sample['confidence']:.3f}")
        if len(wrong_samples) > 5:
            print(f"  ... 还有 {len(wrong_samples)-5} 个错误样本")
    
    # 保存结果
    summary_result = {
        'platform': 'Python',
        'framework': 'ONNX Runtime Python API',
        'test_type': 'real_mnist_data',
        'data_source': 'MNIST test set subset',
        'results': results,
        'summary': {
            'accuracy': accuracy,
            'average_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'fps': 1000/avg_time,
            'total_samples': len(results),
            'correct_predictions': correct_predictions,
            'wrong_predictions': len(wrong_samples)
        }
    }
    
    # 确保results目录存在
    os.makedirs('../results', exist_ok=True)
    
    with open('../results/python_inference_mnist_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary_result, f, indent=2, ensure_ascii=False)
    
    print("结果已保存到: ../results/python_inference_mnist_results.json")
    
    return summary_result

if __name__ == "__main__":
    results = test_python_inference_mnist()
    
    if results:
        print("\n✅ Python推理测试完成")
        print("现在可以运行C/C++推理测试进行对比")
    else:
        print("❌ Python推理测试失败") 