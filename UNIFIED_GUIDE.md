# 统一版本 C/C++ 推理部署指南

本指南介绍如何使用统一版本的C/C++代码进行跨平台MNIST推理部署。

## 📖 概述

统一版本通过单一源码支持macOS和Android平台，使用预处理器宏实现平台适配，大大简化了跨平台部署的复杂性。

### 🔧 核心文件

- **`inference/c_inference_unified.c`** - 统一的C语言推理代码
- **`inference/cpp_inference_unified.cpp`** - 统一的C++推理代码
- **`build/CMakeLists_unified.txt`** - 统一的CMake配置文件
- **`build/build_unified.sh`** - 统一的编译脚本
- **`build/deploy_and_test_unified.sh`** - 统一的部署测试脚本
- **`run_all_platforms_unified.sh`** - 全平台自动化测试脚本

## 🚀 快速开始

### 1. 完整流程（推荐）

运行全自动化测试流程：

```bash
# 在项目根目录执行
./run_all_platforms_unified.sh
```

这将自动完成：
- ✅ 环境检查
- ✅ 模型训练和导出
- ✅ 测试数据生成
- ✅ Python推理测试
- ✅ macOS统一版本编译和测试
- ✅ Android统一版本编译和测试
- ✅ 跨平台性能分析
- ✅ 报告生成

### 2. 分步执行

#### Step 1: 编译统一版本

```bash
cd build

# 编译macOS版本
./build_unified.sh macos

# 编译Android版本（需要连接Android设备）
./build_unified.sh android
```

#### Step 2: 运行测试

```bash
# 测试macOS版本
./deploy_and_test_unified.sh macos

# 测试Android版本（需要连接Android设备）
./deploy_and_test_unified.sh android
```

## 🔧 环境要求

### macOS 环境
- **CMake** 3.18+
- **ONNX Runtime** 库（通过brew安装或手动编译）
- **C/C++编译器** (Xcode Command Line Tools)

```bash
# 安装依赖
brew install cmake onnxruntime
```

### Android 环境
- **Android NDK** (推荐版本 r21+)
- **预编译的ONNX Runtime Android库**
- **连接的Android设备** (启用开发者模式和USB调试)

```bash
# 设置环境变量
export ANDROID_NDK_HOME="/opt/homebrew/share/android-ndk"
```

## 📁 输出文件

### 可执行文件
- **macOS**: `inference/cpp_inference_unified`, `inference/c_inference_unified`
- **Android**: `android_executables/arm64-v8a/cpp_inference_unified`, `android_executables/arm64-v8a/c_inference_unified`

### 结果文件
- **`results/macos_unified_cpp_results.txt`** - macOS C++结果
- **`results/macos_unified_c_results.txt`** - macOS C结果
- **`results/android_unified_cpp_results.txt`** - Android C++结果
- **`results/android_unified_c_results.txt`** - Android C结果

### 报告文件
- **`results/unified_cross_platform_report.md`** - 跨平台性能分析报告
- **`results/unified_cross_platform_analysis.png`** - 可视化性能图表
- **`results/unified_deployment_report.md`** - 部署测试报告

## 🏗️ 架构设计

### 平台适配机制

统一版本使用预处理器宏实现平台差异处理：

```c
#ifdef __ANDROID__
    #define MODEL_PATH "models/mnist_model.onnx"
    #define RESULTS_PATH "results/android_unified_c_results.txt"
    #define PLATFORM_NAME "Android"
#else
    #define MODEL_PATH "../../models/mnist_model.onnx"
    #define RESULTS_PATH "../../results/macos_unified_c_results.txt"
    #define PLATFORM_NAME "macOS"
#endif
```

### 数据加载适配

```c
#ifdef __ANDROID__
// Android数据加载函数
int load_labels_from_metadata(LabelMap* label_map);
float* generate_sample_image(int sample_id);
#else
// macOS数据加载函数
int load_mnist_test_data(MNISTTestData* test_data);
#endif
```

## 📊 性能对比

统一版本能够：

1. **维护一致性** - 相同的算法逻辑保证结果一致性
2. **降低维护成本** - 单一源码减少重复开发
3. **简化部署** - 自动化脚本处理平台差异
4. **性能优化** - 针对不同平台的编译优化

### 典型性能表现

| 平台 | 语言 | 平均推理时间 | FPS | 准确率 |
|------|------|--------------|-----|--------|
| macOS | C++ | ~2-5ms | 200-500 | >95% |
| macOS | C | ~2-5ms | 200-500 | >95% |
| Android | C++ | ~5-15ms | 60-200 | >95% |
| Android | C | ~5-15ms | 60-200 | >95% |

*实际性能取决于硬件配置*

## 🛠️ 自定义配置

### 修改模型路径

编辑统一源码中的路径定义：

```c
#ifdef __ANDROID__
    #define MODEL_PATH "your_android_model_path"
#else
    #define MODEL_PATH "your_macos_model_path"
#endif
```

### 添加新平台支持

1. 在统一源码中添加新的平台宏判断
2. 在CMakeLists中添加平台特定配置
3. 更新编译脚本支持新平台

```c
#ifdef __YOUR_PLATFORM__
    // 平台特定配置
#endif
```

## 🐛 故障排除

### 常见问题

1. **ONNX Runtime未找到**
   ```bash
   # macOS
   brew install onnxruntime
   
   # 或设置路径
   export ONNXRUNTIME_ROOT=/path/to/onnxruntime
   ```

2. **Android NDK未找到**
   ```bash
   export ANDROID_NDK_HOME=/path/to/android-ndk
   ```

3. **Android设备连接问题**
   ```bash
   adb devices  # 检查设备连接
   adb kill-server && adb start-server  # 重启ADB
   ```

4. **编译错误**
   - 检查CMake版本是否>=3.18
   - 确认ONNX Runtime库路径正确
   - 验证Android NDK版本兼容性

### 调试模式

启用详细输出：

```bash
# 编译时启用调试模式
./build_unified.sh android --verbose

# 运行时查看详细日志
./deploy_and_test_unified.sh android --debug
```

## 📚 相关文档

- [QUICK_START.md](QUICK_START.md) - 快速开始指南
- [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) - 执行指南
- [android_cross_compile_guide.md](android_cross_compile_guide.md) - Android交叉编译指南

## 🤝 贡献

欢迎提交问题和改进建议！统一版本的设计目标是简化跨平台部署，让AI模型能够轻松运行在不同设备上。

## 📄 许可证

本项目采用开源许可证，详情请查看LICENSE文件。 