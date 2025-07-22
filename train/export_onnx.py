import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np
import os
from train_model import MNISTNet
import torchvision.transforms as transforms
import torchvision

def export_to_onnx():
    """将PyTorch模型导出为ONNX格式"""
    print("开始导出ONNX模型...")
    
    # 加载训练好的模型
    print("加载PyTorch模型...")
    model = MNISTNet()
    model.load_state_dict(torch.load('../models/mnist_model.pth', map_location='cpu', weights_only=True))
    model.eval()
    
    # 创建示例输入（MNIST图像大小为28x28）
    print("创建示例输入...")
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # 导出ONNX模型
    onnx_path = '../models/mnist_model.onnx'
    print(f"导出ONNX模型到: {onnx_path}")
    
    torch.onnx.export(
        model,                          # 要导出的模型
        dummy_input,                    # 示例输入
        onnx_path,                      # 输出路径
        export_params=True,             # 导出参数
        opset_version=11,               # ONNX算子集版本
        do_constant_folding=True,       # 常量折叠优化
        input_names=['input'],          # 输入节点名称
        output_names=['output'],        # 输出节点名称
        dynamic_axes={                  # 动态维度（支持不同batch size）
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # 验证ONNX模型
    print("验证ONNX模型...")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型验证通过")
    except Exception as e:
        print(f"✗ ONNX模型验证失败: {e}")
        return None
    
    # 显示模型信息
    print("\n=== ONNX模型信息 ===")
    print(f"ONNX版本: {onnx.version.version}")
    print(f"算子集版本: {onnx_model.opset_import[0].version}")
    print(f"输入节点: {[input.name for input in onnx_model.graph.input]}")
    print(f"输出节点: {[output.name for output in onnx_model.graph.output]}")
    
    # 测试ONNX Runtime推理
    print("\n测试ONNX Runtime推理...")
    ort_session = onnxruntime.InferenceSession(onnx_path)
    
    # 准备测试数据
    test_input = dummy_input.numpy()
    
    # PyTorch推理
    print("执行PyTorch推理...")
    with torch.no_grad():
        pytorch_output = model(dummy_input)
    
    # ONNX Runtime推理
    print("执行ONNX Runtime推理...")
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_output = ort_session.run(None, ort_inputs)
    
    # 比较结果
    print("比较推理结果...")
    try:
        np.testing.assert_allclose(
            pytorch_output.numpy(), 
            ort_output[0], 
            rtol=1e-03, 
            atol=1e-05
        )
        print("✓ PyTorch和ONNX Runtime推理结果一致")
    except AssertionError as e:
        print(f"✗ 推理结果不一致: {e}")
        return None
    
    # 使用真实MNIST数据测试
    print("\n使用真实MNIST数据测试...")
    test_with_real_data(model, ort_session)
    
    print(f"\n✓ ONNX模型导出成功: {onnx_path}")
    return onnx_path

def test_with_real_data(pytorch_model, onnx_session):
    """使用真实MNIST数据测试两个模型的一致性"""
    
    # 加载测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='../data', train=False, download=False, transform=transform
    )
    
    # 测试5个样本
    print("测试样本:")
    for i in range(5):
        image, true_label = test_dataset[i]
        input_batch = image.unsqueeze(0)  # 添加batch维度
        
        # PyTorch推理
        with torch.no_grad():
            pytorch_output = pytorch_model(input_batch)
            pytorch_pred = pytorch_output.argmax(dim=1).item()
            pytorch_prob = torch.exp(pytorch_output).max().item()
        
        # ONNX推理
        ort_inputs = {onnx_session.get_inputs()[0].name: input_batch.numpy()}
        ort_output = onnx_session.run(None, ort_inputs)
        onnx_logits = ort_output[0]
        onnx_probs = np.exp(onnx_logits) / np.sum(np.exp(onnx_logits))
        onnx_pred = np.argmax(onnx_probs)
        onnx_prob = np.max(onnx_probs)
        
        # 比较结果
        match = "✓" if pytorch_pred == onnx_pred else "✗"
        print(f"  样本{i}: 真实={true_label}, PyTorch={pytorch_pred}({pytorch_prob:.3f}), "
              f"ONNX={onnx_pred}({onnx_prob:.3f}) {match}")



if __name__ == "__main__":
    onnx_path = export_to_onnx()
    
    if onnx_path:
        print(f"\n✅ ONNX模型导出成功!")
        print(f"📁 模型文件: {onnx_path}")
        print(f"📊 可以使用Netron等工具外部查看模型结构")
        print(f"\n🎉 导出完成，可以继续下一步Python推理测试！") 