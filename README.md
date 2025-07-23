# 🚀 DL2C: 深度学习跨平台部署框架

[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Android-blue)](https://github.com/your-repo/DL2C)
[![Language](https://img.shields.io/badge/Language-Python%20%7C%20C%2B%2B%20%7C%20C-green)](https://github.com/your-repo/DL2C)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20ONNX%20Runtime-orange)](https://github.com/your-repo/DL2C)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **完整的深度学习模型跨平台部署解决方案**  
> 从PyTorch训练到移动端推理的端到端工作流

## 📋 概述

DL2C (Deep Learning to C) 是一个完整的跨平台AI模型部署框架，专注于MNIST手写数字识别任务。项目展示了从模型训练、优化、导出到多平台高性能推理的完整工作流，使用统一的源代码支持macOS和Android平台。

### ✨ 核心特性

- 🎯 **完整工作流**: PyTorch训练 → ONNX导出 → 跨平台部署
- 🔥 **多语言支持**: Python、C++、C三种语言实现
- 📱 **跨平台部署**: macOS本地 + Android移动端
- ⚡ **高性能推理**: 最高6600+ FPS推理速度
- 📊 **智能分析**: 自动化性能对比和可视化分析
- 🛠️ **一键部署**: 全自动化编译、部署、测试流程
- 🔧 **统一架构**: 单一源码支持多平台，降低维护成本

### 🏆 性能基准

| 平台/语言 | FPS | 推理时间 | 准确率 | 部署复杂度 |
|-----------|-----|----------|--------|------------|
| **macOS C++** | 6662 | 0.150ms | 99.0% | 低 |
| **macOS C** | 2614 | 0.383ms | 99.0% | 低 |
| **macOS Python** | 2518 | 0.397ms | 99.0% | 极低 |
| **Android C** | 2386 | 0.420ms | 99.0% | 中 |
| **Android C++** | 2355 | 0.425ms | 99.0% | 中 |

**🔥 关键发现**:
- **最大性能差距**: macOS C++ 比 Android C++ 快 **2.8倍**
- **算法一致性**: 所有平台准确率均为 **99.0%**
- **移动端性能**: Android 达到 **2300+ FPS**，满足实时应用需求

## 🚀 快速开始

### ⚡ 一键运行

# 1. 全流程自动化 - 完整的训练、编译、部署、测试（15-30分钟）
./run_all_platforms.sh

# 2. 查看结果 - 6子图综合性能分析
open results/cross_platform_analysis.png
cat results/cross_platform_report.md
```

### 📋 环境要求

#### 基础依赖
```bash
# Python环境
pip install torch torchvision onnx onnxruntime numpy matplotlib Pillow netron

# macOS工具链
brew install cmake ninja android-ndk pkg-config openjdk@11
brew install --cask android-platform-tools  #adb

# onnxruntime 运行时
brew install onnxruntime
```

#### Android设备
- Android设备连接并开启USB调试
- 验证连接：`adb devices`

#### ONNX Runtime Android 版本编译

这是整个流程中最复杂的部分，需要从源码编译 ONNX Runtime 的 Android 版本。

##### 1. 克隆 ONNX Runtime 源码

```bash
cd /Users/mintisan/Workplaces  # 或您的工作目录
git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
```

##### 2. 编译 Android 版本 (预计 30-60 分钟)

```bash
# 设置环境变量
export ANDROID_NDK_HOME="/opt/homebrew/share/android-ndk"
export ANDROID_SDK_ROOT="$HOME/android-sdk"
export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"

# 开始编译 (这是一个耗时的过程)
./build.sh --android \
  --android_abi arm64-v8a \
  --android_api 21 \
  --android_sdk_path $ANDROID_SDK_ROOT \
  --android_ndk_path $ANDROID_NDK_HOME \
  --config Release \
  --cmake_extra_defines onnxruntime_USE_KLEIDIAI=OFF
```

##### 3. 验证编译结果

```bash
# 检查生成的库文件
ls -la build/Android/Release/libonnxruntime_*.a

# 应该看到以下文件:
# libonnxruntime_session.a
# libonnxruntime_providers.a  
# libonnxruntime_framework.a
# libonnxruntime_graph.a
# 等等...
```

##### 4. 注意事项

- **⏱️ 编译时间**: 首次编译需要30-60分钟，取决于机器性能
- **💾 磁盘空间**: 编译过程需要约5-10GB的磁盘空间
- **🔧 依赖工具**: 确保已安装cmake、ninja、git等工具
- **📱 SDK配置**: Android SDK和NDK路径需要正确设置

##### 5. 编译故障排除

```bash
# 如果编译失败，可以尝试清理后重新编译
./build.sh --clean
rm -rf build/Android

# 确认环境变量设置
echo "NDK: $ANDROID_NDK_HOME"
echo "SDK: $ANDROID_SDK_ROOT"
```

**⚠️ 重要**: 只有成功编译ONNX Runtime Android版本后，才能运行本项目的Android测试。如果编译遇到问题，请参考[ONNX Runtime官方文档](https://onnxruntime.ai/docs/build/android.html)。

## 📁 项目结构

```
DL2C/
├── 🧠 train/                       # 模型训练模块
│   ├── train_model.py              # PyTorch模型训练
│   ├── quantize_model.py           # 模型量化优化
│   └── export_onnx.py              # ONNX格式导出
├── ⚡ inference/                   # 跨平台推理实现
│   ├── python_inference.py        # Python版本（开发友好）
│   ├── cpp_inference.cpp          # C++版本（高性能）
│   └── c_inference.c              # C版本（最大兼容性）
├── 🔨 build/                       # 编译配置和构建输出
│   ├── CMakeLists.txt              # 统一的CMake配置
│   ├── build_android/             # Android构建目录
│   ├── build_macos/               # macOS构建目录
│   ├── onnxruntime-android-arm64-v8a/ # Android ONNX Runtime
│   └── onnxruntime-osx-arm64-1.16.0/  # macOS ONNX Runtime
├── 🔧 build.sh                     # 统一的编译脚本
├── 📱 deploy_and_test.sh           # 自动部署测试脚本
├── 📊 models/                      # 训练好的模型
│   └── mnist_model.onnx           # ONNX格式模型
├── 📈 results/                     # 性能分析结果
│   ├── *_c_results.txt            # C语言推理结果
│   ├── *_cpp_results.txt          # C++推理结果
│   ├── python_inference_results.json # Python推理结果
│   ├── cross_platform_analysis.png # 可视化性能图表
│   └── cross_platform_report.md   # 详细分析报告
├── 📦 test_data/                   # 测试数据，来自data_loader.py
├── 🚀 run_all_platforms.sh         # 一键完整测试
├── data_loader.py                 # 测试数据生成
├── android_executables/           # Android可执行文件
└── 📋 README.md                    # 项目说明（本文件）
```

## 🏗️ 统一架构设计

### 核心理念

项目采用**统一源码架构**，通过预处理器宏实现平台适配，大大简化跨平台部署：

```c
#ifdef __ANDROID__
    #define MODEL_PATH "/data/local/tmp/mnist_onnx/models/mnist_model.onnx"
    #define RESULTS_PATH "/data/local/tmp/mnist_onnx/results/android_c_results.txt"
    #define PLATFORM_NAME "Android"
#else
    #define MODEL_PATH "../models/mnist_model.onnx"
    #define RESULTS_PATH "../results/macos_c_results.txt"
    #define PLATFORM_NAME "macOS"
#endif
```

### 平台适配机制

- **路径适配**: 自动处理不同平台的文件路径差异
- **数据加载**: 统一的数据加载接口，平台特定的实现
- **编译配置**: CMake自动检测平台并应用相应配置
- **部署流程**: 单一脚本处理不同平台的部署差异

## 📖 详细使用指南

### 🎯 分步执行模式

如果不想运行完整流程，可以分步执行：

#### 1. 训练阶段
```bash
cd train
python train_model.py        # 训练模型，下载数据
python quantize_model.py     # 量化优化
python export_onnx.py        # 导出ONNX
netron # 查看 model 结构
```

#### 2. 本地推理测试
```bash
python data_loader.py       # 生成推理测试数据
```
```bash
cd inference
python python_inference.py  # Python推理基准
```

#### 3. 编译跨平台版本
```bash
# 编译macOS版本
./build.sh macos

# 编译Android版本（需连接Android设备）
./build.sh android
```

#### 4. 部署测试
```bash
# 测试macOS版本
./deploy_and_test.sh macos

# 测试Android版本（需连接Android设备）
./deploy_and_test.sh android
```

### 🔧 自定义配置

#### 修改测试规模
```python
# 编辑 data_loader.py
num_samples = 1000  # 默认100，可改为更大规模测试
```

#### 启用详细日志
```bash
# 编译时启用详细输出
./build.sh android --verbose

# 测试时查看详细日志
./deploy_and_test.sh android --debug
```

#### 模型路径自定义
```c
// 编辑推理源码，修改模型路径
#define MODEL_PATH "your_custom_model_path"
```

## 🛠️ 技术栈

### 核心框架
- **🧠 训练**: PyTorch 2.4+
- **🔄 转换**: ONNX 标准格式
- **⚡ 推理**: ONNX Runtime C/C++ API
- **🔨 构建**: CMake + Ninja

### 开发环境
- **💻 平台**: macOS (Apple Silicon), Android (ARM64)
- **🗣️ 语言**: Python 3.8+, C++17, C99
- **📱 移动端**: Android NDK 25+, API Level 21+

### 分析工具
- **📊 可视化**: Matplotlib, NumPy
- **📈 分析**: 自研跨平台性能分析系统
- **🔍 诊断**: 智能环境检测和故障排除

## 🚨 故障排除

### 常见问题解决

#### ❌ Android编译失败
```bash
# 检查NDK配置
echo $ANDROID_NDK_HOME
ls $ANDROID_NDK_HOME/toolchains/llvm/prebuilt/

# 重新安装NDK
brew reinstall android-ndk
```

#### ❌ ONNX Runtime未找到
```bash
# macOS安装
brew install onnxruntime

# 或设置环境变量
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
```

#### ❌ 设备连接问题
```bash
# 重启ADB服务
adb kill-server && adb start-server
adb devices

# 确认USB调试已开启
```

#### ❌ 图表显示问题
图表中文显示乱码时，脚本会自动切换到英文模式。如需中文显示，请确保系统安装了中文字体。

### 🔍 获取帮助

1. **查看日志**: 运行脚本时注意错误输出信息
2. **重置环境**: 删除`build/build_*`目录后重新编译
3. **性能分析**: 查看`results/`目录下的详细报告

## 📊 项目特色

### 🔬 算法一致性验证
- **跨平台准确率**: 所有平台均达到99.0%准确率
- **数值一致性**: ONNX标准保证算法完全一致
- **自动验证**: 智能检测平台间性能差异

### ⚡ 性能优化亮点
- **极致优化**: macOS C++达到6600+ FPS
- **移动端高效**: Android设备2300+ FPS满足实时需求
- **内存高效**: 静态链接部署，无外部依赖

### 🛠️ 开发体验
- **自动化工具链**: 一键完成训练→编译→部署→测试
- **智能故障诊断**: 详细的错误检测和解决建议
- **可视化分析**: 6子图综合性能分析报告
- **统一代码库**: 降低维护成本，提高开发效率

## 📚 扩展学习

### 🎓 进阶功能
- **🔬 算法扩展**: 尝试其他深度学习模型（ResNet、Transformer等）
- **⚡ 性能优化**: GPU加速、量化优化、算子融合
- **📱 移动端开发**: 集成到Android Studio项目
- **🌐 Web部署**: 使用ONNX.js部署到浏览器

### 🏗️ 工程应用
- **CI/CD集成**: 将测试流程集成到持续集成系统
- **生产监控**: 添加推理性能监控和告警
- **模型管理**: 版本控制和A/B测试框架
- **多模型支持**: 扩展为通用模型部署平台

## 🤝 贡献指南

我们欢迎社区贡献！

### 🛠️ 开发流程
1. **Fork** 本仓库
2. **创建** 特性分支：`git checkout -b feature/amazing-feature`
3. **提交** 改动：`git commit -m 'Add amazing feature'`
4. **推送** 分支：`git push origin feature/amazing-feature`
5. **提交** Pull Request

### 📝 贡献方向
- 🐛 Bug修复和性能优化
- 📚 文档改进和翻译
- 🔧 新平台支持（iOS、Web等）
- 🧪 新模型和算法集成

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- **PyTorch团队**: 提供优秀的深度学习框架
- **ONNX社区**: 推动AI模型标准化
- **Microsoft**: 开发高性能ONNX Runtime
- **贡献者们**: 感谢所有为项目做出贡献的开发者

---

<div align="center">

**🌟 如果这个项目对你有帮助，请给个Star！🌟**

**🚀 开始你的跨平台AI部署之旅！**

</div> 