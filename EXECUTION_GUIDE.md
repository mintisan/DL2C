# 🚀 跨平台 MNIST 推理完整执行指南

## 📋 概述

本指南将帮你运行完整的5个平台配置测试：
- **本地**: Python、C、C++
- **Android**: C、C++

预期总时间：15-30分钟（首次运行）

## 🔧 前置要求

### 环境依赖
```bash
# Python依赖
pip install torch torchvision onnx onnxruntime numpy matplotlib Pillow

# macOS工具
brew install cmake ninja android-ndk openjdk@11

# 环境变量
export ANDROID_NDK_HOME="/opt/homebrew/share/android-ndk"
export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"
```

### Android设备
- 连接Android设备并开启USB调试
- 验证连接：`adb devices`

## 🎯 方法一：自动化执行（推荐）

### 一键运行所有测试
```bash
# 给脚本执行权限
chmod +x run_all_platforms.sh

# 运行完整测试流程
./run_all_platforms.sh
```

自动化脚本将：
1. ✅ 检查环境和依赖
2. 🧠 训练模型并导出ONNX（如需要）
3. 📊 生成测试数据（如需要）
4. 🐍 运行Python推理
5. 🔨 编译本地C/C++版本
6. ⚡ 运行C/C++推理
7. 📱 编译Android版本
8. 🤖 运行Android推理
9. 📈 生成性能分析报告

## 🔍 方法二：手动分步执行

### 步骤1: 模型准备
```bash
# 训练MNIST模型
cd train
python train_model.py

# 导出ONNX模型
python export_onnx.py
cd ..
```

### 步骤2: 生成测试数据
```bash
# 生成真实MNIST测试数据
python mnist_data_loader.py
```

### 步骤3: 本地Python推理
```bash
cd inference
python python_inference_mnist.py
cd ..
```

### 步骤4: 本地C/C++推理
```bash
# 编译本地版本
cd build
mkdir -p build_macos && cd build_macos
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4

# 运行C++推理
./bin/mnist_inference_cpp_mnist

# 运行C推理
./bin/mnist_inference_c_mnist
cd ../..
```

### 步骤5: Android跨平台推理
```bash
# 编译Android版本
cd build
export ANDROID_NDK_HOME="/opt/homebrew/share/android-ndk"
export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"
./build_android_real_onnx.sh

# 部署并测试
./deploy_and_test_real_onnx.sh
cd ..
```

### 步骤6: 生成分析报告
```bash
# 运行跨平台性能分析
python android_cross_platform_analysis.py
```

## 📊 查看结果

### 生成的文件
```bash
results/
├── python_inference_mnist_results.json     # Python结果
├── c_inference_mnist_results.json          # C语言结果  
├── cpp_inference_mnist_results.json        # C++结果
├── android_real_onnx_results.txt           # Android C++结果
├── android_real_onnx_c_results.txt         # Android C结果
├── comprehensive_cross_platform_report.md  # 详细报告
└── comprehensive_cross_platform_analysis.png # 可视化图表
```

### 查看方式
```bash
# 查看详细报告
cat results/comprehensive_cross_platform_report.md

# 查看图表 (macOS)
open results/comprehensive_cross_platform_analysis.png

# 查看图表 (Linux)
xdg-open results/comprehensive_cross_platform_analysis.png
```

## 🎯 预期结果

### 性能排行榜
1. **🥇 macOS C++**: ~6600 FPS (最快)
2. **🥈 macOS C**: ~2600 FPS
3. **🥉 macOS Python**: ~2500 FPS  
4. **4️⃣ Android C**: ~2400 FPS
5. **5️⃣ Android C++**: ~2350 FPS

### 准确率
- **所有平台**: 99.0% 准确率（算法一致性）

## 🛠️ 故障排除

### 常见问题

#### 1. Python依赖问题
```bash
# 重新安装依赖
pip install --upgrade torch onnx onnxruntime numpy matplotlib
```

#### 2. Android编译失败
```bash
# 检查NDK路径
echo $ANDROID_NDK_HOME
ls $ANDROID_NDK_HOME/toolchains/llvm/prebuilt/

# 重新安装NDK
brew reinstall android-ndk
```

#### 3. Android设备连接问题
```bash
# 检查设备连接
adb devices

# 重启ADB服务
adb kill-server && adb start-server

# 检查USB调试是否开启
adb shell getprop ro.debuggable
```

#### 4. 本地编译失败
```bash
# 清理编译缓存
cd build/build_macos
make clean
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

#### 5. 图表字体问题
```bash
# 修改配置使用英文
# 编辑 android_cross_platform_analysis.py
# 设置 FORCE_ENGLISH = True
```

## 📈 自定义配置

### 修改测试样本数
```python
# 编辑 mnist_data_loader.py
num_samples = 1000  # 默认100，可改为更多
```

### 启用中文图表
```python
# 编辑 android_cross_platform_analysis.py  
FORCE_ENGLISH = False  # 尝试使用中文（需字体支持）
```

### 选择性运行测试
```bash
# 仅运行本地测试
python inference/python_inference_mnist.py
cd build/build_macos && ./bin/mnist_inference_cpp_mnist

# 仅运行Android测试  
cd build && ./deploy_and_test_real_onnx.sh

# 仅生成分析报告
python android_cross_platform_analysis.py
```

## 🎉 成功标志

完成后你将看到：
- ✅ 5个配置的推理结果文件
- ✅ 综合性能分析报告
- ✅ 6子图可视化图表
- ✅ 详细的性能排行榜
- ✅ 跨平台算法一致性验证

## 💡 后续扩展

- 🔧 集成到CI/CD流程
- 📱 开发Android App
- ⚡ 添加GPU加速支持
- 🌐 部署为Web服务
- 🧪 测试更多AI模型

---
*最后更新: 2025-07-22* 