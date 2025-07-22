# Android 跨平台编译快速参考指南

## 🚀 一键设置脚本

为了简化复杂的环境配置，这里提供一键设置脚本：

### macOS 环境一键配置

```bash
#!/bin/bash
# 保存为 setup_android_env.sh

echo "=== DL2C Android 跨平台编译环境配置 ==="

# 1. 安装基础工具
echo "安装基础工具..."
brew install cmake ninja git wget curl openjdk@11 android-ndk

# 2. 设置环境变量
echo "配置环境变量..."
export ANDROID_NDK_HOME="/opt/homebrew/share/android-ndk"
export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"

# 3. 安装 Android SDK
echo "安装 Android SDK..."
mkdir -p ~/android-sdk && cd ~/android-sdk
curl -O https://dl.google.com/android/repository/commandlinetools-mac-9477386_latest.zip
unzip commandlinetools-mac-9477386_latest.zip
mkdir -p cmdline-tools/latest
mv cmdline-tools/* cmdline-tools/latest/ 2>/dev/null || true

export ANDROID_SDK_ROOT=$HOME/android-sdk
export PATH=$PATH:$ANDROID_SDK_ROOT/cmdline-tools/latest/bin

# 4. 接受许可
echo "接受 Android SDK 许可..."
yes | sdkmanager --licenses

echo "✅ 环境配置完成！"
echo "请运行以下命令将环境变量添加到 shell 配置："
echo "echo 'export ANDROID_NDK_HOME=\"/opt/homebrew/share/android-ndk\"' >> ~/.zshrc"
echo "echo 'export ANDROID_SDK_ROOT=\"$HOME/android-sdk\"' >> ~/.zshrc"
echo "echo 'export PATH=\"/opt/homebrew/opt/openjdk@11/bin:\$PATH\"' >> ~/.zshrc"
echo "echo 'export PATH=\"\$PATH:\$ANDROID_SDK_ROOT/cmdline-tools/latest/bin\"' >> ~/.zshrc"
```

## ⚡ 快速编译流程

### 简化版本（推荐新手）

```bash
# 1. 克隆项目
git clone <project-url> && cd DL2C

# 2. 准备模型和数据
python train/train_model.py
python train/export_onnx.py
python mnist_data_loader.py

# 3. 交叉编译
cd build
./build_android_simple.sh

# 4. 部署测试
./deploy_and_test_simple.sh
```

### 完整 ONNX 版本（工业级）

```bash
# 1. 编译 ONNX Runtime (30-60 分钟)
cd ～/Workplaces
git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
./build.sh --android --android_abi arm64-v8a --android_api 21 \
  --android_sdk_path $ANDROID_SDK_ROOT \
  --android_ndk_path $ANDROID_NDK_HOME \
  --config Release

# 2. 编译项目
cd ～/Workplaces/DL2C/build
./build_android_real_onnx.sh

# 3. 部署测试
./deploy_and_test_real_onnx.sh
```

## 🔧 故障排除快速检查

### 环境检查脚本

```bash
#!/bin/bash
# 保存为 check_env.sh

echo "=== 环境检查 ==="

# 检查必需工具
for tool in cmake ninja git java adb; do
    if command -v $tool >/dev/null 2>&1; then
        echo "✅ $tool: $(which $tool)"
    else
        echo "❌ $tool: 未安装"
    fi
done

# 检查 Android 工具
if [ -d "$ANDROID_NDK_HOME" ]; then
    echo "✅ Android NDK: $ANDROID_NDK_HOME"
else
    echo "❌ Android NDK: 未配置 ANDROID_NDK_HOME"
fi

if [ -d "$ANDROID_SDK_ROOT" ]; then
    echo "✅ Android SDK: $ANDROID_SDK_ROOT"
else
    echo "❌ Android SDK: 未配置 ANDROID_SDK_ROOT"
fi

# 检查设备连接
DEVICE_COUNT=$(adb devices | grep -c "device$" || echo "0")
if [ "$DEVICE_COUNT" -gt 0 ]; then
    echo "✅ Android 设备: $DEVICE_COUNT 个已连接"
    adb devices
else
    echo "⚠️ Android 设备: 未连接"
fi

# 检查 ONNX Runtime 编译结果
if [ -f "/Users/$(whoami)/Workplaces/onnxruntime/build/Android/Release/libonnxruntime_session.a" ]; then
    echo "✅ ONNX Runtime Android: 已编译"
else
    echo "⚠️ ONNX Runtime Android: 未编译或路径不正确"
fi
```

## 📊 性能基准参考

### 预期性能数据

| 平台 | 语言 | 推理时间 | FPS | 相对性能 |
|------|------|----------|-----|----------|
| macOS M1 | Python | ~1.2ms | 833 | 1x |
| macOS M1 | C++ | ~0.03ms | 33,333 | 40x |
| macOS M1 | C | ~0.006ms | 166,667 | 200x |
| Android ARM64 | C++ (简化) | ~0.013ms | 76,923 | 92x |
| Android ARM64 | C (简化) | ~0.009ms | 111,111 | 133x |
| Android ARM64 | C++ (ONNX) | ~0.5ms | 2,000 | 2.4x |

### 文件大小参考

| 版本 | 可执行文件大小 | 依赖库总大小 | 内存占用 |
|------|----------------|--------------|----------|
| 简化版本 | ~50KB | 0 (无外部依赖) | ~1MB |
| ONNX 版本 | ~20MB | ~100MB+ | ~50MB |

## 🎯 最佳实践建议

### 开发阶段

1. **先简化版本**: 验证交叉编译流程
2. **逐步复杂化**: 理解每个组件的作用
3. **版本控制**: 使用 Git 管理配置文件

### 生产部署

1. **选择合适版本**: 根据性能需求选择
2. **静态链接**: 减少运行时依赖
3. **大小优化**: 使用 `-Oz` 优化文件大小
4. **性能测试**: 在目标设备上充分测试

### 团队协作

1. **Docker 容器**: 统一开发环境
2. **CI/CD 流水线**: 自动化编译和测试
3. **文档维护**: 及时更新环境配置
4. **错误收集**: 建立问题反馈机制

## 🔗 相关资源

### 官方文档
- [ONNX Runtime Build Guide](https://onnxruntime.ai/docs/build/android.html)
- [Android NDK Guide](https://developer.android.com/ndk/guides)
- [CMake Cross Compiling](https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html)

### 社区资源
- [ONNX Runtime Issues](https://github.com/microsoft/onnxruntime/issues)
- [Android NDK Samples](https://github.com/android/ndk-samples)

### 性能优化
- [ARM NEON 优化指南](https://developer.arm.com/documentation/den0018/a)
- [Android 性能分析工具](https://developer.android.com/studio/profile)

---

**注意**: 此指南基于 macOS Apple Silicon + Android ARM64 环境编写，其他平台可能需要调整配置。 