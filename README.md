# 🚀 DL2C: 深度学习跨平台部署框架

[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Android-blue)](https://github.com/your-repo/DL2C)
[![Language](https://img.shields.io/badge/Language-Python%20%7C%20C%2B%2B%20%7C%20C-green)](https://github.com/your-repo/DL2C)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20ONNX%20Runtime-orange)](https://github.com/your-repo/DL2C)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **完整的深度学习模型跨平台部署解决方案**  
> 从PyTorch训练到移动端推理的端到端工作流

## 📋 概述

DL2C (Deep Learning to C) 是一个完整的跨平台AI模型部署框架，专注于MNIST手写数字识别任务。项目展示了从模型训练、优化、导出到多平台高性能推理的完整工作流。

### ✨ 核心特性

- 🎯 **完整工作流**: PyTorch训练 → ONNX导出 → 跨平台部署
- 🔥 **多语言支持**: Python、C++、C三种语言实现
- 📱 **跨平台部署**: macOS本地 + Android移动端
- ⚡ **高性能推理**: 最高6600+ FPS推理速度
- 📊 **智能分析**: 自动化性能对比和可视化分析
- 🛠️ **一键部署**: 全自动化编译、部署、测试流程

### 🏆 性能基准

| 平台/语言 | FPS | 推理时间 | 准确率 | 部署复杂度 |
|-----------|-----|----------|--------|------------|
| **macOS C++** | 6662 | 0.150ms | 99.0% | 低 |
| **macOS C** | 2614 | 0.383ms | 99.0% | 低 |
| **macOS Python** | 2518 | 0.397ms | 99.0% | 极低 |
| **Android C** | 2386 | 0.420ms | 99.0% | 中 |
| **Android C++** | 2355 | 0.425ms | 99.0% | 中 |

## 🚀 快速开始

### ⚡ 一键运行（推荐）

```bash
# 1. 检查环境状态
./quick_check.sh

# 2. 运行完整测试流程（15-30分钟）
./run_all_platforms.sh

# 3. 查看分析结果
open results/comprehensive_cross_platform_analysis.png
```

### 📋 环境要求

#### 基础依赖
```bash
# Python依赖
pip install torch torchvision onnx onnxruntime numpy matplotlib Pillow

# macOS工具（使用Homebrew）
brew install cmake ninja android-ndk openjdk@11
```

#### Android设备
- Android设备连接并开启USB调试
- 验证连接：`adb devices`

### 🎯 验证安装

```bash
# 快速环境检查
./quick_check.sh

# 如果显示"5/5个配置已完成"，说明环境完美！
```

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

## 📁 项目结构

```
DL2C/
├── 🧠 train/                       # 模型训练
│   ├── train_model.py              # PyTorch模型训练
│   ├── quantize_model.py           # 模型量化优化
│   └── export_onnx.py              # ONNX格式导出
├── ⚡ inference/                   # 多语言推理实现
│   ├── python_inference_mnist.py  # Python版本（开发友好）
│   ├── cpp_inference_mnist.cpp    # C++版本（高性能）
│   ├── c_inference_mnist.c        # C版本（最大兼容性）
│   └── android_real_onnx_*         # Android专用版本
├── 🔨 build/                       # 编译和部署配置
│   ├── CMakeLists.txt              # 本地编译配置
│   ├── CMakeLists_android_real.txt # Android交叉编译
│   ├── build_android_real_onnx.sh  # Android编译脚本
│   └── deploy_and_test_real_onnx.sh # 自动部署测试
├── 📊 models/                      # 训练好的模型
├── 📈 results/                     # 性能分析结果
├── 🚀 run_all_platforms.sh         # 一键完整测试
├── 🔍 quick_check.sh               # 快速环境检查
├── android_cross_platform_analysis.py # 性能分析工具
├── 📖 QUICK_START.md               # 快速开始指南
├── 📚 EXECUTION_GUIDE.md           # 详细执行步骤
└── 📋 README.md                    # 项目说明（本文件）
```

## 📖 详细使用指南

### 🎯 使用场景

#### 1. 学习研究
```bash
# 快速验证概念
./quick_check.sh
python inference/python_inference_mnist.py
```

#### 2. 性能基准测试
```bash
# 完整性能测试
./run_all_platforms.sh
python android_cross_platform_analysis.py
```

#### 3. 生产部署
```bash
# 专门的平台测试
cd build && ./deploy_and_test_real_onnx.sh  # Android
cd build_macos && ./bin/mnist_inference_cpp_mnist  # 本地高性能
```

### 🔧 自定义配置

#### 修改测试规模
```python
# 编辑 mnist_data_loader.py
num_samples = 1000  # 默认100，可改为更大规模测试
```

#### 启用中文图表
```python
# 编辑 android_cross_platform_analysis.py
FORCE_ENGLISH = False  # 启用中文（需要字体支持）
```

#### 模型优化选项
```python
# 编辑 quantize_model.py
# 尝试不同的量化策略以获得更好的性能/精度平衡
```

## 🛠️ 技术栈

### 核心框架
- **🧠 训练**: PyTorch 2.4+
- **🔄 转换**: ONNX 格式标准
- **⚡ 推理**: ONNX Runtime
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

### 常见问题

#### ❌ Android编译失败
```bash
# 检查NDK配置
echo $ANDROID_NDK_HOME
ls $ANDROID_NDK_HOME/toolchains/llvm/prebuilt/

# 重新安装NDK
brew reinstall android-ndk
```

#### ❌ 设备连接问题
```bash
# 重启ADB服务
adb kill-server && adb start-server
adb devices

# 确认USB调试已开启
```

#### ❌ Python依赖缺失
```bash
# 重新安装依赖
pip install --upgrade torch onnx onnxruntime numpy matplotlib
```

#### ❌ 图表字体问题
图表显示乱码时，脚本会自动切换到英文模式。如需中文，确保系统安装了中文字体。

### 🔍 获取帮助

1. **运行诊断**: `./quick_check.sh`
2. **查看详细日志**: 运行脚本时注意输出的错误信息
3. **重置环境**: 删除`build/build_*`目录重新编译
4. **查看文档**: `cat EXECUTION_GUIDE.md`

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

[📖 快速开始](QUICK_START.md) • [📚 详细指南](EXECUTION_GUIDE.md) • [🐛 问题反馈](https://github.com/your-repo/DL2C/issues) • [💡 功能建议](https://github.com/your-repo/DL2C/discussions)

</div> 