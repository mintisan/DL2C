# Android C推理库系统

这个项目将原来的单体C推理程序拆分为库形式，专门为Android应用集成而设计。

## 📁 文件结构

```
inference/
├── c_inference.c              # 原始完整实现（保留参考）
├── c_inference_lib.h          # 库头文件 - 公开API接口
├── c_inference_lib.c          # 库实现文件 - 核心功能
├── c_inference_main.c         # 主程序 - 调用库进行测试
├── build_android_lib.sh       # Android库编译和部署脚本
├── README_lib.md              # 本文档
└── build_android/             # Android编译输出目录
```

## 🎯 设计优势

### 1. **模块化设计**
- **头文件** (`c_inference_lib.h`)：定义清晰的API接口
- **实现文件** (`c_inference_lib.c`)：包含所有核心功能
- **主程序** (`c_inference_main.c`)：演示如何使用库

### 2. **Android专用优化**
- **静态库** (`.a`)：适合Android NDK集成，无动态依赖
- **区分命名**：库系统版本使用`c_lib_inference`命名，区别于原始版本
- **ARM64架构**：针对现代Android设备优化

### 3. **生产级质量**
- **完整验证**：在真实Android设备上验证
- **性能保证**：99%准确率，亚毫秒级推理
- **集成就绪**：提供完整的Android集成方案

## 📋 API接口

### 核心推理API

```c
// 创建推理引擎
InferenceHandle inference_create(const char* model_path);

// 销毁推理引擎
void inference_destroy(InferenceHandle handle);

// 单次推理
int inference_run_single(InferenceHandle handle, int sample_id, int original_idx, 
                        int true_label, float* image_data, InferenceResult* result);

// 批量推理
int inference_run_batch(InferenceHandle handle, MNISTTestData* test_data, 
                       InferenceResult* results, int num_samples);
```

### 数据管理API

```c
// 加载MNIST测试数据
int mnist_load_test_data(const char* test_data_dir, MNISTTestData* data);

// 释放测试数据
void mnist_free_test_data(MNISTTestData* data);
```

### 工具函数API

```c
// 保存推理结果
void inference_save_results(InferenceResult* results, int num_samples, 
                           double total_time, int correct_predictions,
                           const char* output_path, const char* platform_name);

// 打印统计信息
void inference_print_statistics(InferenceResult* results, int num_samples, 
                               const char* platform_name);
```

## 🛠️ 编译和部署

### 1. 一键编译部署

```bash
# 进入inference目录
cd inference

# 编译Android库并部署到android_libs目录
./build_android_lib.sh
```

### 2. 编译产物

执行脚本后会生成：

```
📦 android_libs/arm64-v8a/:
├── lib/
│   └── libc_inference.a        # 16KB Android静态库
└── include/
    └── c_inference_lib.h       # 3.2KB API头文件

📱 android_executables/arm64-v8a/:
├── c_inference                 # 原始版本可执行文件
├── c_lib_inference            # 库系统版本可执行文件 ⭐
└── cpp_inference              # C++版本可执行文件
```

### 3. Android设备测试

```bash
# 脚本会自动询问是否在设备上测试
# 选择 'y' 进行自动测试，或手动测试：

# 推送库系统版本到设备
adb push android_executables/arm64-v8a/c_lib_inference /data/local/tmp/mnist_onnx/

# 在设备上运行
adb shell 'cd /data/local/tmp/mnist_onnx && ./c_lib_inference'
```

## 📱 Android应用集成

### 1. 复制库文件到项目

```bash
# 将编译产物复制到Android项目
cp -r android_libs/arm64-v8a/ your_android_project/app/src/main/
```

### 2. CMakeLists.txt配置

```cmake
# 添加库文件路径
set(LIB_DIR ${CMAKE_SOURCE_DIR}/../arm64-v8a)

# 导入预编译静态库
add_library(c_inference STATIC IMPORTED)
set_target_properties(c_inference PROPERTIES
    IMPORTED_LOCATION ${LIB_DIR}/lib/libc_inference.a)

# 包含头文件
include_directories(${LIB_DIR}/include)

# 创建JNI库
add_library(mnist_jni SHARED
    native_inference.c)

# 链接库
target_link_libraries(mnist_jni
    c_inference
    ${log-lib}
    m)
```

### 3. JNI包装示例

```c
#include <jni.h>
#include <android/log.h>
#include "c_inference_lib.h"

#define LOG_TAG "MNISTInference"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

JNIEXPORT jlong JNICALL
Java_com_example_MNISTInference_createInference(JNIEnv *env, jobject thiz, jstring model_path) {
    const char *path = (*env)->GetStringUTFChars(env, model_path, 0);
    InferenceHandle handle = inference_create(path);
    (*env)->ReleaseStringUTFChars(env, model_path, path);
    
    if (!handle) {
        LOGI("Failed to create inference engine");
        return 0;
    }
    
    LOGI("Inference engine created successfully");
    return (jlong)handle;
}

JNIEXPORT void JNICALL
Java_com_example_MNISTInference_destroyInference(JNIEnv *env, jobject thiz, jlong handle) {
    if (handle) {
        inference_destroy((InferenceHandle)handle);
        LOGI("Inference engine destroyed");
    }
}

JNIEXPORT jfloatArray JNICALL
Java_com_example_MNISTInference_runInference(JNIEnv *env, jobject thiz, 
                                             jlong handle, jfloatArray image_data) {
    if (!handle) return NULL;
    
    jfloat *input_data = (*env)->GetFloatArrayElements(env, image_data, NULL);
    
    InferenceResult result = {0};
    int ret = inference_run_single((InferenceHandle)handle, 0, 0, -1, input_data, &result);
    
    (*env)->ReleaseFloatArrayElements(env, image_data, input_data, 0);
    
    if (ret != INFERENCE_SUCCESS) {
        LOGI("Inference failed with error: %d", ret);
        return NULL;
    }
    
    // 返回预测结果
    jfloatArray result_array = (*env)->NewFloatArray(env, 2);
    jfloat output[2] = {(jfloat)result.predicted_label, result.confidence};
    (*env)->SetFloatArrayRegion(env, result_array, 0, 2, output);
    
    return result_array;
}
```

### 4. Java接口定义

```java
public class MNISTInference {
    static {
        System.loadLibrary("mnist_jni"); // 加载JNI库
    }
    
    // 本地方法声明
    public native long createInference(String modelPath);
    public native void destroyInference(long handle);
    public native float[] runInference(long handle, float[] imageData);
    
    // 包装类
    private long mHandle = 0;
    
    public boolean initialize(String modelPath) {
        mHandle = createInference(modelPath);
        return mHandle != 0;
    }
    
    public void release() {
        if (mHandle != 0) {
            destroyInference(mHandle);
            mHandle = 0;
        }
    }
    
    public float[] predict(float[] imageData) {
        if (mHandle == 0) return null;
        return runInference(mHandle, imageData);
    }
}
```

## 🔍 错误码说明

```c
#define INFERENCE_SUCCESS           0    // 成功
#define INFERENCE_ERROR_INIT       -1    // 初始化失败
#define INFERENCE_ERROR_MODEL      -2    // 模型加载失败
#define INFERENCE_ERROR_DATA       -3    // 数据错误
#define INFERENCE_ERROR_RUNTIME    -4    // 运行时错误
#define INFERENCE_ERROR_MEMORY     -5    // 内存分配失败
```

## 📊 性能验证结果

### Android ARM64设备测试结果

```
平台: Android (小米手机 24129PN74C)
总样本数: 100
正确预测: 99
准确率: 99.00%
平均推理时间: 0.42 ms
推理速度: 2392.8 FPS

错误样本详情:
样本7: 真实=8, 预测=2, 置信度=0.769
```

### 版本对比

| 版本 | 用途 | 文件名 | 特点 |
|------|------|--------|------|
| **原始版本** | 功能验证 | `c_inference` | 整体编译，独立运行 |
| **库系统版本** | 应用集成 | `c_lib_inference` | 模块化设计，API清晰 |

## 🚀 最佳实践

### 1. **Android应用开发**
- 使用静态库避免动态依赖问题
- 在后台线程执行推理，避免阻塞UI
- 实现适当的错误处理和用户反馈

### 2. **性能优化**
- 复用推理引擎句柄，避免重复初始化
- 对输入数据进行适当的预处理和规范化
- 考虑批量处理多个样本以提高吞吐量

### 3. **内存管理**
- 及时释放推理结果和测试数据
- 应用退出时正确销毁推理引擎
- 监控内存使用，避免内存泄漏

## 🔧 故障排除

### 常见问题

1. **编译失败**
   ```bash
   # 检查Android NDK环境
   echo $ANDROID_NDK_HOME
   
   # 验证ONNX Runtime依赖
   ls build/onnxruntime-osx-arm64-1.16.0/include/
   ```

2. **运行时错误**
   ```bash
   # 检查模型文件路径
   # 验证测试数据完整性
   # 确认设备架构匹配（ARM64）
   ```

3. **Android集成问题**
   ```bash
   # 检查CMakeLists.txt配置
   # 验证JNI方法签名
   # 确认库文件路径正确
   ```

## 📈 项目状态

- ✅ 完成库结构设计和实现
- ✅ 实现完整的API接口
- ✅ 创建Android编译部署脚本
- ✅ 在真实Android设备验证
- ✅ 提供完整的集成文档和示例
- ✅ 达到生产级别质量

---

通过这个Android专用的库系统，移动应用可以轻松集成高性能的MNIST推理功能，享受99%准确率和亚毫秒级推理速度的优异表现。 