import onnxruntime as ort
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
from PIL import Image
import time
import json
import os

class PythonONNXInference:
    """Python ONNX推理类"""
    
    def __init__(self, model_path):
        """初始化ONNX推理引擎"""
        print(f"加载ONNX模型: {model_path}")
        
        # 创建ONNX Runtime会话（推理使用CPU以确保兼容性）
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        print("🖥️  推理使用CPU执行，确保跨平台兼容性")
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        
        print(f"输入名称: {self.input_name}")
        print(f"输入形状: {self.input_shape}")
        print(f"输出名称: {self.output_name}")
        print(f"输出形状: {self.output_shape}")
        
    def preprocess(self, image):
        """预处理图像数据"""
        if isinstance(image, Image.Image):
            # PIL图像处理
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            tensor = transform(image)
            return tensor.unsqueeze(0).numpy()
        elif isinstance(image, np.ndarray):
            # NumPy数组处理
            if image.ndim == 2:  # 单通道图像
                image = image.reshape(1, 1, 28, 28)
            image = image.astype(np.float32) / 255.0
            # 标准化
            image = (image - 0.1307) / 0.3081
            return image
        elif isinstance(image, torch.Tensor):
            # PyTorch张量处理
            if image.dim() == 3:  # [C, H, W] -> [1, C, H, W]
                image = image.unsqueeze(0)
            elif image.dim() == 2:  # [H, W] -> [1, 1, H, W]
                image = image.unsqueeze(0).unsqueeze(0)
            return image.numpy().astype(np.float32)
        else:
            return image
    
    def postprocess(self, output):
        """后处理输出结果"""
        # 获取ONNX输出 (形状通常是 [batch_size, num_classes])
        logits = output[0]
        
        # 如果是批次输出，取第一个样本
        if logits.ndim == 2:
            logits = logits[0]  # 从 [1, 10] 变为 [10]
        
        # 应用softmax获得概率分布
        exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性
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
    
    def inference(self, input_data):
        """执行推理"""
        start_time = time.time()
        
        # 预处理
        processed_input = self.preprocess(input_data)
        
        # 运行推理
        ort_inputs = {self.input_name: processed_input}
        ort_outputs = self.session.run([self.output_name], ort_inputs)
        
        # 后处理
        result = self.postprocess(ort_outputs)
        
        # 添加推理时间
        inference_time = time.time() - start_time
        result['inference_time_ms'] = inference_time * 1000
        
        return result
    
    def batch_inference(self, input_batch):
        """批量推理"""
        results = []
        for i, input_data in enumerate(input_batch):
            result = self.inference(input_data)
            result['batch_index'] = i
            results.append(result)
        return results

def test_python_inference():
    """测试Python ONNX推理"""
    print("=== Python ONNX推理测试 ===")
    
    # 检查模型文件是否存在
    model_path = '../models/mnist_model.onnx'
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print("请先运行训练和ONNX导出脚本")
        return None
    
    # 初始化推理引擎
    inference_engine = PythonONNXInference(model_path)
    
    # 加载测试数据
    print("\n加载MNIST测试数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='../data', train=False, download=False, transform=transform
    )
    
    # 测试单个样本
    print("\n=== 单样本推理测试 ===")
    test_samples = 10
    results = []
    
    for i in range(test_samples):
        image, true_label = test_dataset[i]
        
        # 执行推理
        result = inference_engine.inference(image)
        result['true_label'] = int(true_label)
        result['sample_id'] = i
        
        results.append(result)
        
        # 显示结果
        status = "✓" if result['predicted_class'] == true_label else "✗"
        print(f"样本 {i:2d}: 真实={true_label}, 预测={result['predicted_class']}, "
              f"置信度={result['confidence']:.4f}, 时间={result['inference_time_ms']:.2f}ms {status}")
    
    # 计算统计信息
    print("\n=== 性能统计 ===")
    correct = sum(1 for r in results if r['predicted_class'] == r['true_label'])
    accuracy = correct / len(results)
    avg_time = np.mean([r['inference_time_ms'] for r in results])
    std_time = np.std([r['inference_time_ms'] for r in results])
    
    print(f"准确率: {accuracy:.2%} ({correct}/{len(results)})")
    print(f"平均推理时间: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"推理速度: {1000/avg_time:.1f} FPS")
    
    # 测试批量推理
    print("\n=== 批量推理测试 ===")
    batch_size = 32
    batch_data = [test_dataset[i][0] for i in range(batch_size)]
    batch_labels = [test_dataset[i][1] for i in range(batch_size)]
    
    start_time = time.time()
    batch_results = inference_engine.batch_inference(batch_data)
    batch_time = time.time() - start_time
    
    batch_correct = sum(1 for i, r in enumerate(batch_results) 
                       if r['predicted_class'] == batch_labels[i])
    batch_accuracy = batch_correct / len(batch_results)
    
    print(f"批量大小: {batch_size}")
    print(f"批量准确率: {batch_accuracy:.2%}")
    print(f"批量总时间: {batch_time*1000:.2f} ms")
    print(f"平均单样本时间: {batch_time*1000/batch_size:.2f} ms")
    print(f"批量推理速度: {batch_size/batch_time:.1f} FPS")
    
    # 保存结果
    print("\n保存推理结果...")
    os.makedirs('../results', exist_ok=True)
    
    summary_result = {
        'model_path': model_path,
        'test_samples': test_samples,
        'results': results,
        'summary': {
            'accuracy': accuracy,
            'average_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'fps': 1000/avg_time,
            'total_samples': len(results),
            'correct_predictions': correct
        },
        'batch_test': {
            'batch_size': batch_size,
            'batch_accuracy': batch_accuracy,
            'batch_total_time_ms': batch_time * 1000,
            'batch_avg_time_per_sample_ms': batch_time * 1000 / batch_size,
            'batch_fps': batch_size / batch_time
        }
    }
    
    with open('../results/python_inference_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary_result, f, indent=2, ensure_ascii=False)
    
    print("结果已保存到: ../results/python_inference_results.json")
    
    return summary_result

def test_different_inputs():
    """测试不同类型的输入"""
    print("\n=== 测试不同输入类型 ===")
    
    model_path = '../models/mnist_model.onnx'
    inference_engine = PythonONNXInference(model_path)
    
    # 1. 测试PIL图像
    print("1. 测试PIL图像输入...")
    # 创建一个简单的测试图像
    test_image_pil = Image.new('L', (28, 28), 128)  # 灰色图像
    result_pil = inference_engine.inference(test_image_pil)
    print(f"PIL图像推理结果: {result_pil['predicted_class']}, 置信度: {result_pil['confidence']:.4f}")
    
    # 2. 测试NumPy数组
    print("2. 测试NumPy数组输入...")
    test_image_np = np.random.rand(28, 28) * 255
    result_np = inference_engine.inference(test_image_np)
    print(f"NumPy数组推理结果: {result_np['predicted_class']}, 置信度: {result_np['confidence']:.4f}")
    
    # 3. 测试PyTorch张量
    print("3. 测试PyTorch张量输入...")
    test_image_torch = torch.randn(1, 28, 28)
    result_torch = inference_engine.inference(test_image_torch)
    print(f"PyTorch张量推理结果: {result_torch['predicted_class']}, 置信度: {result_torch['confidence']:.4f}")

if __name__ == "__main__":
    # 运行主要测试
    results = test_python_inference()
    
    if results:
        # 测试不同输入类型
        test_different_inputs()
        
        print("\n=== Python推理测试完成 ===")
        print("接下来可以运行C++推理测试进行对比")
    else:
        print("Python推理测试失败，请检查模型文件") 