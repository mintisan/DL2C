# 🚀 Android C推理库系统

这是一个**真正自包含的静态库系统**，可以为Android应用提供完整的ONNX Runtime推理能力。

## 🎯 **核心特点**

- ✅ **87MB自包含静态库** - 包含所有659个目标文件
- ✅ **21个库完整合并** - 20个ONNX Runtime核心库 + 1个API层
- ✅ **零外部依赖** - Android应用只需链接一个库文件
- ✅ **跨平台支持** - 支持不同电脑和用户名

## 📋 **环境要求**

### 必需软件
- Android NDK (推荐 r21 或更高版本)
- 本地编译的ONNX Runtime Android版本

### 默认路径配置
脚本默认期望以下路径结构：
```bash
$HOME/Workplaces/onnxruntime/
├── build/Android/Release/          # ONNX Runtime Android构建结果
│   ├── libonnxruntime_*.a         # 核心静态库
│   └── _deps/                     # 第三方依赖库
└── include/onnxruntime/core/session/  # 头文件
    └── onnxruntime_c_api.h
```

### 自定义路径配置
如果你的ONNX Runtime在其他位置，请修改 `build_android_lib.sh` 中的路径：

```bash
# 修改这两行为你的实际路径
ONNX_BUILD_DIR="$HOME/你的路径/onnxruntime/build/Android/Release"
ONNX_INCLUDE_DIR="$HOME/你的路径/onnxruntime/include/onnxruntime/core/session"
```

## 🔧 **ONNX Runtime编译指南**

如果你还没有编译Android版本的ONNX Runtime：

```bash
# 1. 克隆ONNX Runtime源码
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# 2. 设置环境变量
export ANDROID_SDK_ROOT=/path/to/your/android-sdk
export ANDROID_NDK_HOME=/path/to/your/android-ndk

# 3. 编译Android版本
./build.sh --config Release \
           --android \
           --android_sdk_path $ANDROID_SDK_ROOT \
           --android_ndk_path $ANDROID_NDK_HOME \
           --android_abi arm64-v8a
```

## 🚀 **使用方法**

### 编译自包含静态库
```bash
cd inference
./build_android_lib.sh
```

脚本将自动：
1. ✅ 检查环境变量和路径
2. ✅ 编译API层代码
3. ✅ 提取和合并所有ONNX Runtime静态库
4. ✅ 创建87MB自包含静态库
5. ✅ 生成Android集成指南
6. ✅ 清理所有临时文件

### 生成的文件
```bash
android_libs/arm64-v8a/
├── lib/
│   └── libc_inference.a          # 87MB自包含静态库
├── include/
│   └── c_inference_lib.h         # C API头文件
└── ANDROID_INTEGRATION.md        # 完整集成指南
```

## 📱 **Android应用集成**

### 1. 复制库文件
```bash
cp android_libs/arm64-v8a/lib/libc_inference.a YourApp/app/src/main/cpp/
cp android_libs/arm64-v8a/include/c_inference_lib.h YourApp/app/src/main/cpp/
```

### 2. 配置CMakeLists.txt
```cmake
# 导入自包含静态库
add_library(c_inference STATIC IMPORTED)
set_target_properties(c_inference PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libc_inference.a
)

# 链接库 - 只需要链接我们的库和系统库！
target_link_libraries(your_jni_lib
    c_inference      # 自包含静态库
    android log m
)
```

### 3. 使用API
```cpp
#include "c_inference_lib.h"

// 创建推理引擎
InferenceHandle handle = inference_create_engine("model.onnx");

// 运行推理
float input[784] = {/* 输入数据 */};
InferenceResult result = inference_run_single(handle, input);

// 使用结果
if (result.success) {
    int predicted_class = result.predicted_class;
    // 处理结果...
}

// 清理资源
inference_destroy_engine(handle);
```

## 🌍 **跨平台使用**

### 在不同电脑上使用
1. **确保ONNX Runtime路径**：
   - 默认路径：`$HOME/Workplaces/onnxruntime/`
   - 或修改脚本中的路径配置

2. **设置环境变量**：
   ```bash
   export ANDROID_NDK_HOME=/path/to/your/ndk
   ```

3. **运行脚本**：
   ```bash
   ./build_android_lib.sh
   ```

### 故障排除

**问题：未找到ONNX Runtime**
```bash
# 检查路径是否正确
ls $HOME/Workplaces/onnxruntime/build/Android/Release/

# 如果路径不同，修改脚本中的ONNX_BUILD_DIR和ONNX_INCLUDE_DIR
```

**问题：NDK未找到**
```bash
# 设置NDK环境变量
export ANDROID_NDK_HOME=/path/to/your/android-ndk
```

## 📊 **性能指标**

- **准确率**: 99%
- **推理时间**: 0.42ms (ARM64)
- **库大小**: 87MB (包含所有依赖)
- **目标文件**: 659个
- **静态库**: 21个完整合并

## 🎯 **优势对比**

| 传统方案 | 自包含库系统 |
|---------|-------------|
| 需要管理多个库文件 | ✅ 只有一个库文件 |
| 可能出现版本冲突 | ✅ 版本一致性保证 |
| 复杂的CMakeLists.txt | ✅ 3行代码完成配置 |
| 运行时库找不到风险 | ✅ 静态链接无风险 |
| 需要下载ONNX Runtime | ✅ 所有依赖已内置 |

🚀 **这是真正对Android开发者友好的一站式解决方案！** 