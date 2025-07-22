User: (1) 调研并评估移动端推理框架，核心目标是找到能正确运行模型的方案。优先考察提供纯 C 语言接口 (C API) 的框架（如 ONNX Runtime C API），其次是 C++ 框架（如 PyTorch Mobile, NCNN）。重点评估其在 ARM 平台对 CNN 和 RNN (LSTM/GRU) 算子的支持完整性与 C/C++ 接口的易用性。
(2) 设计一个标准化的基础部署工作流 (Workflow)，确保模型能成功运行。该流程应优先适配纯 C 语言集成，并兼容 C++ 作为备选方案。工作流需明确定义模型转换、基础 C/C++ 代码集成和功能验证等核心阶段。
(3) 以 ECG R-peak 检测模型为试点案例，实践并验证该基础部署工作流。根据调研结果，优先选择 C 语言方案进行端到端部署，确保模型在不经过任何加速优化的情况下能够正确执行并输出结果。
(4) 【可选步骤】针对已成功部署的框架，调研其模型加速方案。这包括研究量化（如 PTQ/QAT）、剪枝等优化方法，以及如何通过 C/C++ 接口利用 ARM NEON 等硬件加速特性来提升性能。
(5) 明确 LLM 代码助手在工作流中的角色。例如，利用 LLM 生成模型转换脚本、调用 C/C++ API 的基础推理代码、编译脚本（Makefile/CMake），以及在可选的加速阶段生成优化相关的代码。
(6) 针对工作流的每个环节，开发可复用的代码模板和脚本，优先创建 C 语言版本。包括标准的模型加载、预处理、推理和后处理的函数模板，并为 C++ 方案准备相应的模板，构成自动化部署“Agent”的核心。
(7) 编写详细的流程文档，记录最终选择的 C 或 C++ 推理库和技术决策。文档需清晰地区分“基础部署”（必选）和“性能加速”（可选）两个部分，并提供相应的操作指南。
(8) 创建“新模型快速落地清单”，总结并指导用户如何利用已建立的工作流和 LLM 助手，首先完成基础部署，然后根据需要选择性地执行性能优化。

Assistant: 我将为您提供一个完整的移动端推理框架调研与部署工作流设计方案。

## 1. 移动端推理框架调研与评估

### 1.1 框架对比分析

| 框架 | 语言接口 | ARM支持 | CNN支持 | RNN/LSTM支持 | 易用性 | 推荐度 |
|------|----------|---------|---------|---------------|--------|--------|
| ONNX Runtime | C API + C++ | ✅ 优秀 | ✅ 完整 | ✅ 完整 | ⭐⭐⭐⭐⭐ | 🥇 首选 |
| TensorFlow Lite | C API + C++ | ✅ 优秀 | ✅ 完整 | ✅ 完整 | ⭐⭐⭐⭐ | 🥈 推荐 |
| NCNN | C++ only | ✅ 优秀 | ✅ 完整 | ⭐ 有限 | ⭐⭐⭐ | 🥉 备选 |
| PyTorch Mobile | C++ only | ✅ 良好 | ✅ 完整 | ✅ 完整 | ⭐⭐ | 备选 |

### 1.2 详细评估报告

**ONNX Runtime (推荐首选)**
- ✅ 提供纯C API (`onnxruntime_c_api.h`)
- ✅ ARM平台优化良好，支持ARM NEON
- ✅ CNN算子支持完整（Conv2D, BatchNorm, ReLU等）
- ✅ RNN算子支持完整（LSTM, GRU, RNN）
- ✅ 跨平台兼容性好
- ✅ 社区活跃，文档完善

**TensorFlow Lite (次选)**
- ✅ 提供C API (`c_api.h`)
- ✅ ARM平台支持良好
- ✅ 算子支持完整
- ⚠️ C API相对简单，部分功能需要C++

## 2. 标准化基础部署工作流设计

### 2.1 工作流架构图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   模型准备阶段   │ -> │   转换集成阶段   │ -> │   验证部署阶段   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
    ┌────▼────┐              ┌────▼────┐              ┌────▼────┐
    │原始模型  │              │ONNX模型 │              │C/C++应用│
    │训练完成  │              │C API   │              │推理验证 │
    └─────────┘              │集成代码 │              └─────────┘
                             └─────────┘
```

### 2.2 详细工作流步骤

**阶段1: 模型准备**
1. 模型格式检查与转换准备
2. 输入输出维度确认
3. 算子兼容性验证

**阶段2: 转换集成**
1. 模型转换为ONNX格式
2. 生成C语言推理代码
3. 编译配置与依赖管理

**阶段3: 验证部署**
1. 功能正确性验证
2. 基础性能测试
3. 内存使用分析

## 3. ECG R-peak检测模型试点实践

### 3.1 模型转换脚本

```python
# convert_ecg_model.py
import torch
import torch.onnx
import numpy as np

def convert_ecg_model_to_onnx(model_path, onnx_path):
    """将ECG R-peak检测模型转换为ONNX格式"""
    
    # 加载PyTorch模型
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # 定义示例输入 (batch_size=1, channels=1, sequence_length=1000)
    dummy_input = torch.randn(1, 1, 1000)
    
    # 转换为ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['ecg_signal'],
        output_names=['r_peaks'],
        dynamic_axes={
            'ecg_signal': {2: 'sequence_length'},
            'r_peaks': {1: 'num_peaks'}
        }
    )
    
    print(f"模型已转换: {onnx_path}")

if __name__ == "__main__":
    convert_ecg_model_to_onnx("ecg_model.pth", "ecg_model.onnx")
```

### 3.2 C语言推理代码模板

```c
// ecg_inference.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnxruntime_c_api.h"

typedef struct {
    const OrtApi* ort_api;
    OrtEnv* env;
    OrtSession* session;
    OrtMemoryInfo* memory_info;
} ECGInferenceContext;

// 初始化推理上下文
int ecg_init_inference(ECGInferenceContext* ctx, const char* model_path) {
    ctx->ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    // 创建环境
    if (ctx->ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ECGInference", &ctx->env) != ORT_OK) {
        return -1;
    }
    
    // 创建会话选项
    OrtSessionOptions* session_options;
    ctx->ort_api->CreateSessionOptions(&session_options);
    
    // 创建会话
    if (ctx->ort_api->CreateSession(ctx->env, model_path, session_options, &ctx->session) != ORT_OK) {
        return -1;
    }
    
    // 创建内存信息
    ctx->ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ctx->memory_info);
    
    ctx->ort_api->ReleaseSessionOptions(session_options);
    return 0;
}

// ECG信号预处理
void ecg_preprocess(float* input_data, int length) {
    // 归一化处理
    float mean = 0.0f, std = 0.0f;
    
    // 计算均值
    for (int i = 0; i < length; i++) {
        mean += input_data[i];
    }
    mean /= length;
    
    // 计算标准差
    for (int i = 0; i < length; i++) {
        std += (input_data[i] - mean) * (input_data[i] - mean);
    }
    std = sqrt(std / length);
    
    // 标准化
    for (int i = 0; i < length; i++) {
        input_data[i] = (input_data[i] - mean) / std;
    }
}

// 执行推理
int ecg_run_inference(ECGInferenceContext* ctx, float* input_data, int input_length, 
                      float** output_data, int* output_length) {
    
    // 预处理
    ecg_preprocess(input_data, input_length);
    
    // 创建输入tensor
    int64_t input_shape[] = {1, 1, input_length};
    OrtValue* input_tensor = NULL;
    
    ctx->ort_api->CreateTensorWithDataAsOrtValue(
        ctx->memory_info,
        input_data,
        input_length * sizeof(float),
        input_shape,
        3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    );
    
    // 设置输入输出名称
    const char* input_names[] = {"ecg_signal"};
    const char* output_names[] = {"r_peaks"};
    
    // 运行推理
    OrtValue* output_tensor = NULL;
    OrtStatus* status = ctx->ort_api->Run(
        ctx->session,
        NULL,
        input_names,
        (const OrtValue* const*)&input_tensor,
        1,
        output_names,
        1,
        &output_tensor
    );
    
    if (status != ORT_OK) {
        return -1;
    }
    
    // 获取输出数据
    float* output_ptr;
    ctx->ort_api->GetTensorMutableData(output_tensor, (void**)&output_ptr);
    
    // 获取输出维度
    OrtTensorTypeAndShapeInfo* output_info;
    ctx->ort_api->GetTensorTypeAndShape(output_tensor, &output_info);
    
    size_t output_count;
    ctx->ort_api->GetTensorShapeElementCount(output_info, &output_count);
    
    // 分配输出内存并复制数据
    *output_data = (float*)malloc(output_count * sizeof(float));
    memcpy(*output_data, output_ptr, output_count * sizeof(float));
    *output_length = (int)output_count;
    
    // 清理资源
    ctx->ort_api->ReleaseValue(input_tensor);
    ctx->ort_api->ReleaseValue(output_tensor);
    ctx->ort_api->ReleaseTensorTypeAndShapeInfo(output_info);
    
    return 0;
}

// 清理资源
void ecg_cleanup_inference(ECGInferenceContext* ctx) {
    if (ctx->session) ctx->ort_api->ReleaseSession(ctx->session);
    if (ctx->memory_info) ctx->ort_api->ReleaseMemoryInfo(ctx->memory_info);
    if (ctx->env) ctx->ort_api->ReleaseEnv(ctx->env);
}

// 主函数示例
int main() {
    ECGInferenceContext ctx;
    
    // 初始化
    if (ecg_init_inference(&ctx, "ecg_model.onnx") != 0) {
        printf("初始化失败\n");
        return -1;
    }
    
    // 模拟ECG数据
    int input_length = 1000;
    float* input_data = (float*)malloc(input_length * sizeof(float));
    
    // 生成模拟数据
    for (int i = 0; i < input_length; i++) {
        input_data[i] = sin(2.0 * M_PI * i / 100.0) + 0.1 * (rand() / (float)RAND_MAX);
    }
    
    // 执行推理
    float* output_data;
    int output_length;
    
    if (ecg_run_inference(&ctx, input_data, input_length, &output_data, &output_length) == 0) {
        printf("推理成功，检测到 %d 个R峰\n", output_length);
        
        // 打印结果
        for (int i = 0; i < output_length; i++) {
            printf("R峰位置 %d: %.2f\n", i, output_data[i]);
        }
        
        free(output_data);
    } else {
        printf("推理失败\n");
    }
    
    // 清理
    free(input_data);
    ecg_cleanup_inference(&ctx);
    
    return 0;
}
```

### 3.3 编译配置 (CMakeLists.txt)

```cmake
cmake_minimum_required(VERSION 3.10)
project(ECGInference)

set(CMAKE_C_STANDARD 99)

# 查找ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_c_api.h
    PATHS /usr/local/include/onnxruntime
          /opt/onnxruntime/include
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS /usr/local/lib
          /opt/onnxruntime/lib
)

# 添加可执行文件
add_executable(ecg_inference ecg_inference.c)

# 链接库
target_include_directories(ecg_inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(ecg_inference ${ONNXRUNTIME_LIB} m)

# ARM优化标志
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    target_compile_options(ecg_inference PRIVATE -mfpu=neon -O3)
endif()
```

## 4. 模型加速方案调研

### 4.1 量化优化

```python
# quantization.py
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_ecg_model(model_path, quantized_path):
    """动态量化ECG模型"""
    quantize_dynamic(
        model_path,
        quantized_path,
        weight_type=QuantType.QUInt8
    )
    print(f"量化模型保存至: {quantized_path}")

# 使用示例
quantize_ecg_model("ecg_model.onnx", "ecg_model_quantized.onnx")
```

### 4.2 ARM NEON加速代码

```c
// neon_optimization.c
#ifdef __ARM_NEON
#include <arm_neon.h>

void vectorized_preprocess(float* data, int length) {
    // 使用NEON指令进行向量化预处理
    int simd_length = length - (length % 4);
    
    for (int i = 0; i < simd_length; i += 4) {
        float32x4_t vec = vld1q_f32(&data[i]);
        // 向量化操作
        vec = vmulq_n_f32(vec, 0.5f);  // 示例：缩放
        vst1q_f32(&data[i], vec);
    }
    
    // 处理剩余元素
    for (int i = simd_length; i < length; i++) {
        data[i] *= 0.5f;
    }
}
#endif
```

## 5. LLM代码助手集成方案

### 5.1 LLM助手角色定义

```python
# llm_assistant.py
class DeploymentAssistant:
    """部署助手类"""
    
    def generate_conversion_script(self, model_info):
        """生成模型转换脚本"""
        prompt = f"""
        请生成将{model_info['framework']}模型转换为ONNX的Python脚本。
        模型类型: {model_info['type']}
        输入维度: {model_info['input_shape']}
        输出维度: {model_info['output_shape']}
        """
        return self.query_llm(prompt)
    
    def generate_inference_code(self, model_spec):
        """生成C语言推理代码"""
        prompt = f"""
        请生成使用ONNX Runtime C API的推理代码。
        要求:
        1. 包含完整的错误处理
        2. 支持{model_spec['input_type']}输入
        3. 输出{model_spec['output_type']}结果
        4. 包含预处理和后处理函数
        """
        return self.query_llm(prompt)
    
    def generate_build_script(self, platform_info):
        """生成编译脚本"""
        # 实现编译脚本生成逻辑
        pass
```

## 6. 可复用代码模板库

### 6.1 通用推理模板 (C语言版本)

```c
// inference_template.h
#ifndef INFERENCE_TEMPLATE_H
#define INFERENCE_TEMPLATE_H

#include "onnxruntime_c_api.h"

typedef struct {
    const OrtApi* api;
    OrtEnv* env;
    OrtSession* session;
    OrtMemoryInfo* memory_info;
    char** input_names;
    char** output_names;
    int num_inputs;
    int num_outputs;
} InferenceContext;

// 通用接口
int init_inference_context(InferenceContext* ctx, const char* model_path);
int run_inference(InferenceContext* ctx, void** inputs, void** outputs);
void cleanup_inference_context(InferenceContext* ctx);

// 预处理函数模板
void normalize_input(float* data, int length);
void resize_input(float* input, float* output, int in_w, int in_h, int out_w, int out_h);

// 后处理函数模板
void apply_softmax(float* data, int length);
int find_max_index(float* data, int length);

#endif
```

### 6.2 模型特化模板生成器

```python
# template_generator.py
class TemplateGenerator:
    """代码模板生成器"""
    
    def __init__(self):
        self.c_template = """
// {model_name}_inference.c
#include "inference_template.h"

typedef struct {{
    InferenceContext base;
    // 模型特定配置
    {model_specific_config}
}} {ModelName}Context;

int {model_name}_init({ModelName}Context* ctx, const char* model_path) {{
    // 初始化代码
    {init_code}
}}

int {model_name}_preprocess(float* input, int input_size, float** processed, int* processed_size) {{
    // 预处理代码
    {preprocess_code}
}}

int {model_name}_inference({ModelName}Context* ctx, float* input, float** output) {{
    // 推理代码
    {inference_code}
}}

int {model_name}_postprocess(float* raw_output, int output_size, {output_type}* result) {{
    // 后处理代码
    {postprocess_code}
}}
"""
    
    def generate_model_code(self, model_config):
        """根据配置生成特化代码"""
        return self.c_template.format(**model_config)
```

## 7. 详细流程文档

### 7.1 技术决策文档

**选择ONNX Runtime的原因:**
1. ✅ 提供完整的C API，符合纯C语言要求
2. ✅ ARM平台优化成熟，性能表现优异
3. ✅ 算子支持最全面，特别是RNN系列
4. ✅ 社区活跃，长期维护有保障
5. ✅ 跨平台兼容性最佳

**部署流程分层:**

**基础部署层 (必选)**
- 模型转换与验证
- C API集成与编译
- 功能正确性验证

**性能优化层 (可选)**
- 模型量化与压缩
- 硬件加速利用
- 内存与计算优化

### 7.2 操作指南

**基础部署步骤:**

1. **环境准备**
   ```bash
   # 安装ONNX Runtime
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-aarch64-1.16.0.tgz
   tar -xzf onnxruntime-linux-aarch64-1.16.0.tgz
   export ONNXRUNTIME_ROOT=/path/to/onnxruntime
   ```

2. **模型转换**
   ```bash
   python convert_model.py --input model.pth --output model.onnx
   ```

3. **代码生成**
   ```bash
   python generate_inference_code.py --model model.onnx --output inference.c
   ```

4. **编译部署**
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ./model_inference
   ```

**性能优化步骤:**

1. **模型量化**
   ```bash
   python quantize_model.py --input model.onnx --output model_quantized.onnx
   ```

2. **编译优化**
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_NEON=ON ..
   make
   ```

## 8. 新模型快速落地清单

### 8.1 快速部署检查清单

**阶段1: 模型准备 ✓**
- [ ] 确认模型格式 (PyTorch/TensorFlow/ONNX)
- [ ] 验证模型输入输出维度
- [ ] 检查使用的算子类型
- [ ] 准备测试数据集

**阶段2: 基础部署 ✓**
- [ ] 执行模型转换脚本
- [ ] 生成C语言推理代码
- [ ] 配置编译环境
- [ ] 验证推理结果正确性
- [ ] 测试基础性能指标

**阶段3: 性能优化 (可选)**
- [ ] 评估是否需要量化
- [ ] 测试量化后精度损失
- [ ] 启用硬件加速选项
- [ ] 优化内存使用
- [ ] 性能基准测试

### 8.2 LLM助手使用指南

**代码生成命令:**
```bash
# 生成转换脚本
llm_assistant generate-conversion --model-type pytorch --input-shape "1,1,1000" --output-shape "1,100"

# 生成推理代码
llm_assistant generate-inference --api onnxruntime-c --model-spec config.json

# 生成编译脚本
llm_assistant generate-build --platform arm64 --optimization-level 3
```

**自动化部署命令:**
```bash
# 一键部署
deploy_agent --model model.pth --target arm64 --framework onnxruntime --language c

# 性能优化
optimize_agent --model model.onnx --methods quantization,neon --target arm64
```

### 8.3 故障排除指南

**常见问题解决:**

1. **转换失败**
   - 检查算子兼容性
   - 更新ONNX版本
   - 简化模型结构

2. **编译错误**
   - 验证库路径配置
   - 检查头文件包含
   - 确认链接库版本

3. **运行时错误**
   - 验证输入数据格式
   - 检查内存分配
   - 确认模型文件完整性

4. **性能问题**
   - 启用编译器优化
   - 使用硬件加速
   - 考虑模型量化

这个完整的工作流程确保了从模型训练到移动端部署的标准化、自动化流程，优先保证功能正确性，然后提供可选的性能优化方案。通过LLM助手和代码模板，可以大大简化新模型的部署过程。

User: 以 PyTrorch 训练 MNIST 为例给一个完整的过程【macOS】，给出训练+量化，导出ONNX，Netron 查看模型结构，采用 Python ONNX 推理，以及 C/C++ 推理，以及 NDK 编译成 ARM 手机平台的库文件和跨平台可执行文件，以及通过 adb 导入手机运行对比原始/macOS ONNX 本地推理/手机推理结果对比

Assistant: 我将为您提供一个完整的MNIST模型从训练到移动端部署的端到端流程。

## 完整流程概览

```
训练阶段 -> 量化优化 -> 模型导出 -> 可视化验证 -> Python推理 -> C++推理 -> NDK编译 -> 手机部署 -> 结果对比
```

## 1. 环境准备

### 1.1 macOS环境配置

```bash
# 创建虚拟环境
conda create -n mnist_deploy python=3.9
conda activate mnist_deploy

# 安装依赖
pip install torch torchvision torchaudio
pip install onnx onnxruntime
pip install numpy matplotlib
pip install netron

# 安装Android NDK (如果没有)
brew install android-ndk
export ANDROID_NDK_HOME=/opt/homebrew/share/android-ndk
```

### 1.2 目录结构

```
mnist_project/
├── train/
│   ├── train_model.py
│   ├── quantize_model.py
│   └── export_onnx.py
├── inference/
│   ├── python_inference.py
│   ├── cpp_inference.cpp
│   └── android_inference.cpp
├── build/
│   ├── CMakeLists.txt
│   └── android/
├── models/
│   ├── mnist_model.pth
│   ├── mnist_quantized.pth
│   └── mnist_model.onnx
├── data/
└── results/
```

## 2. PyTorch训练MNIST模型

### 2.1 训练脚本

```python
# train/train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def train_model():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    
    # 训练
    model.train()
    for epoch in range(5):  # 快速训练5个epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    # 测试
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # 保存模型
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/mnist_model.pth')
    torch.save(model, '../models/mnist_model_full.pth')
    print("模型已保存到 ../models/mnist_model.pth")
    
    return model

if __name__ == "__main__":
    trained_model = train_model()
```

## 3. 模型量化

### 3.1 量化脚本

```python
# train/quantize_model.py
import torch
import torch.quantization as quant
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from train_model import MNISTNet

def quantize_model():
    # 加载原始模型
    model = MNISTNet()
    model.load_state_dict(torch.load('../models/mnist_model.pth', map_location='cpu'))
    model.eval()
    
    # 准备校准数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    calibration_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=False, transform=transform
    )
    calibration_loader = DataLoader(calibration_dataset, batch_size=32, shuffle=False)
    
    # 配置量化
    model.qconfig = quant.get_default_qconfig('fbgemm')
    
    # 准备量化
    model_prepared = quant.prepare(model, inplace=False)
    
    # 校准（使用少量数据）
    model_prepared.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(calibration_loader):
            if i >= 10:  # 只使用10个batch进行校准
                break
            model_prepared(data)
    
    # 转换为量化模型
    model_quantized = quant.convert(model_prepared, inplace=False)
    
    # 测试量化后的精度
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in calibration_loader:
            outputs = model_quantized(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'量化后模型精度: {accuracy:.2f}%')
    
    # 保存量化模型
    torch.save(model_quantized.state_dict(), '../models/mnist_quantized.pth')
    torch.save(model_quantized, '../models/mnist_quantized_full.pth')
    print("量化模型已保存")
    
    return model, model_quantized

if __name__ == "__main__":
    original, quantized = quantize_model()
```

## 4. 导出ONNX模型

### 4.1 ONNX导出脚本

```python
# train/export_onnx.py
import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np
from train_model import MNISTNet

def export_to_onnx():
    # 加载模型
    model = MNISTNet()
    model.load_state_dict(torch.load('../models/mnist_model.pth', map_location='cpu'))
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # 导出ONNX
    onnx_path = '../models/mnist_model.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # 验证ONNX模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过")
    
    # 测试ONNX Runtime推理
    ort_session = onnxruntime.InferenceSession(onnx_path)
    
    # 准备测试数据
    test_input = dummy_input.numpy()
    
    # PyTorch推理
    with torch.no_grad():
        pytorch_output = model(dummy_input)
    
    # ONNX Runtime推理
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_output = ort_session.run(None, ort_inputs)
    
    # 比较结果
    np.testing.assert_allclose(
        pytorch_output.numpy(), 
        ort_output[0], 
        rtol=1e-03, 
        atol=1e-05
    )
    print("PyTorch和ONNX Runtime推理结果一致")
    print(f"ONNX模型已导出到: {onnx_path}")
    
    return onnx_path

if __name__ == "__main__":
    onnx_path = export_to_onnx()
```

## 5. Netron查看模型结构

### 5.1 可视化脚本

```python
# visualize_model.py
import netron
import webbrowser
import time

def visualize_onnx_model():
    model_path = './models/mnist_model.onnx'
    
    # 启动Netron服务器
    netron.start(model_path, port=8080, browse=False)
    
    # 自动打开浏览器
    webbrowser.open('http://localhost:8080')
    
    print("Netron服务器已启动: http://localhost:8080")
    print("按Ctrl+C停止服务器")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("服务器已停止")

if __name__ == "__main__":
    visualize_onnx_model()
```

## 6. Python ONNX推理

### 6.1 Python推理脚本

```python
# inference/python_inference.py
import onnxruntime as ort
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
from PIL import Image
import time
import json

class PythonONNXInference:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def preprocess(self, image):
        """预处理图像"""
        if isinstance(image, Image.Image):
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            tensor = transform(image)
            return tensor.unsqueeze(0).numpy()
        elif isinstance(image, np.ndarray):
            # 假设输入已经是28x28的灰度图像
            image = image.astype(np.float32) / 255.0
            image = (image - 0.1307) / 0.3081
            return image.reshape(1, 1, 28, 28)
        else:
            return image
    
    def inference(self, input_data):
        """执行推理"""
        start_time = time.time()
        
        ort_inputs = {self.input_name: input_data}
        ort_outputs = self.session.run([self.output_name], ort_inputs)
        
        inference_time = time.time() - start_time
        
        # 应用softmax获得概率
        logits = ort_outputs[0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        predicted_class = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0][predicted_class]
        
        return {
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'probabilities': probabilities[0].tolist(),
            'inference_time_ms': inference_time * 1000,
            'raw_logits': logits[0].tolist()
        }

def test_python_inference():
    """测试Python推理"""
    print("=== Python ONNX推理测试 ===")
    
    # 初始化推理器
    inference_engine = PythonONNXInference('./models/mnist_model.onnx')
    
    # 加载测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=False, transform=transform
    )
    
    # 测试多个样本
    results = []
    test_samples = 10
    
    for i in range(test_samples):
        image, true_label = test_dataset[i]
        input_data = image.unsqueeze(0).numpy()
        
        result = inference_engine.inference(input_data)
        result['true_label'] = int(true_label)
        result['sample_id'] = i
        
        results.append(result)
        
        print(f"样本 {i}: 真实={true_label}, 预测={result['predicted_class']}, "
              f"置信度={result['confidence']:.4f}, 时间={result['inference_time_ms']:.2f}ms")
    
    # 计算准确率
    correct = sum(1 for r in results if r['predicted_class'] == r['true_label'])
    accuracy = correct / len(results)
    avg_time = np.mean([r['inference_time_ms'] for r in results])
    
    print(f"\n准确率: {accuracy:.2%}")
    print(f"平均推理时间: {avg_time:.2f}ms")
    
    # 保存结果
    with open('./results/python_inference_results.json', 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'accuracy': accuracy,
                'average_inference_time_ms': avg_time,
                'total_samples': len(results)
            }
        }, f, indent=2)
    
    return results

if __name__ == "__main__":
    import os
    os.makedirs('./results', exist_ok=True)
    test_python_inference()
```

## 7. C++推理实现

### 7.1 C++推理代码

```cpp
// inference/cpp_inference.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <memory>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <onnxruntime_cxx_api.h>

class CPPONNXInference {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;

public:
    CPPONNXInference(const std::string& model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "CPPONNXInference"),
          session(env, model_path.c_str(), Ort::SessionOptions{}),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        
        // 获取输入信息
        size_t num_input_nodes = session.GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions{});
            input_names.push_back(input_name.release());
            
            Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            input_shape = input_tensor_info.GetShape();
        }
        
        // 获取输出信息
        size_t num_output_nodes = session.GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions{});
            output_names.push_back(output_name.release());
            
            Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            output_shape = output_tensor_info.GetShape();
        }
        
        std::cout << "模型加载成功" << std::endl;
        std::cout << "输入形状: [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    ~CPPONNXInference() {
        for (auto& name : input_names) {
            delete[] name;
        }
        for (auto& name : output_names) {
            delete[] name;
        }
    }
    
    struct InferenceResult {
        int predicted_class;
        float confidence;
        std::vector<float> probabilities;
        double inference_time_ms;
        std::vector<float> raw_logits;
    };
    
    std::vector<float> preprocess(const std::vector<float>& raw_data) {
        std::vector<float> processed = raw_data;
        
        // 归一化 (假设输入已经是0-1范围)
        const float mean = 0.1307f;
        const float std = 0.3081f;
        
        for (auto& pixel : processed) {
            pixel = (pixel - mean) / std;
        }
        
        return processed;
    }
    
    std::vector<float> softmax(const std::vector<float>& logits) {
        std::vector<float> probabilities(logits.size());
        float max_logit = *std::max_element(logits.begin(), logits.end());
        
        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            probabilities[i] = std::exp(logits[i] - max_logit);
            sum += probabilities[i];
        }
        
        for (auto& prob : probabilities) {
            prob /= sum;
        }
        
        return probabilities;
    }
    
    InferenceResult inference(const std::vector<float>& input_data) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // 预处理
        auto processed_data = preprocess(input_data);
        
        // 创建输入tensor
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            const_cast<float*>(processed_data.data()), 
            processed_data.size(),
            input_shape.data(), 
            input_shape.size()
        );
        
        // 运行推理
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, 
            input_names.data(), 
            &input_tensor, 
            1, 
            output_names.data(), 
            1
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // 获取输出数据
        float* output_data = output_tensors.front().GetTensorMutableData<float>();
        size_t output_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
        
        std::vector<float> logits(output_data, output_data + output_size);
        std::vector<float> probabilities = softmax(logits);
        
        // 找到预测类别
        auto max_it = std::max_element(probabilities.begin(), probabilities.end());
        int predicted_class = std::distance(probabilities.begin(), max_it);
        float confidence = *max_it;
        
        InferenceResult result;
        result.predicted_class = predicted_class;
        result.confidence = confidence;
        result.probabilities = probabilities;
        result.inference_time_ms = duration.count() / 1000.0;
        result.raw_logits = logits;
        
        return result;
    }
};

// 读取MNIST测试数据的简单实现
std::vector<std::pair<std::vector<float>, int>> load_mnist_test_data(const std::string& data_path, int num_samples = 10) {
    // 这里简化实现，实际应该读取MNIST二进制文件
    // 为了演示，我们生成一些随机数据
    std::vector<std::pair<std::vector<float>, int>> data;
    
    for (int i = 0; i < num_samples; ++i) {
        std::vector<float> image(28 * 28);
        for (int j = 0; j < 28 * 28; ++j) {
            image[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        int label = rand() % 10;
        data.emplace_back(image, label);
    }
    
    return data;
}

int main() {
    try {
        std::cout << "=== C++ ONNX推理测试 ===" << std::endl;
        
        // 初始化推理引擎
        CPPONNXInference inference_engine("../models/mnist_model.onnx");
        
        // 加载测试数据
        auto test_data = load_mnist_test_data("../data", 10);
        
        std::vector<CPPONNXInference::InferenceResult> results;
        int correct = 0;
        
        for (size_t i = 0; i < test_data.size(); ++i) {
            const auto& [image, true_label] = test_data[i];
            
            auto result = inference_engine.inference(image);
            results.push_back(result);
            
            if (result.predicted_class == true_label) {
                correct++;
            }
            
            std::cout << "样本 " << i << ": 真实=" << true_label 
                      << ", 预测=" << result.predicted_class
                      << ", 置信度=" << std::fixed << std::setprecision(4) << result.confidence
                      << ", 时间=" << std::setprecision(2) << result.inference_time_ms << "ms"
                      << std::endl;
        }
        
        double accuracy = static_cast<double>(correct) / test_data.size();
        double avg_time = 0.0;
        for (const auto& result : results) {
            avg_time += result.inference_time_ms;
        }
        avg_time /= results.size();
        
        std::cout << std::endl;
        std::cout << "准确率: " << std::setprecision(2) << accuracy * 100 << "%" << std::endl;
        std::cout << "平均推理时间: " << std::setprecision(2) << avg_time << "ms" << std::endl;
        
        // 保存结果到文件
        std::ofstream result_file("../results/cpp_inference_results.json");
        result_file << "{\n";
        result_file << "  \"summary\": {\n";
        result_file << "    \"accuracy\": " << accuracy << ",\n";
        result_file << "    \"average_inference_time_ms\": " << avg_time << ",\n";
        result_file << "    \"total_samples\": " << results.size() << "\n";
        result_file << "  }\n";
        result_file << "}\n";
        result_file.close();
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
```

### 7.2 C++编译配置

```cmake
# build/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(MNISTInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找ONNX Runtime
if(APPLE)
    # macOS路径
    set(ONNXRUNTIME_ROOT_PATH "/usr/local/onnxruntime")
    if(NOT EXISTS ${ONNXRUNTIME_ROOT_PATH})
        # 如果没有安装，下载预编译版本
        set(ONNXRUNTIME_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-x86_64-1.16.0")
    endif()
elseif(ANDROID)
    # Android路径
    set(ONNXRUNTIME_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-android-${ANDROID_ABI}")
endif()

set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.dylib)

if(ANDROID)
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so)
endif()

# 添加可执行文件
add_executable(mnist_inference_cpp ../inference/cpp_inference.cpp)

# 设置包含目录
target_include_directories(mnist_inference_cpp PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})

# 链接库
target_link_libraries(mnist_inference_cpp ${ONNXRUNTIME_LIB})

# Android特定配置
if(ANDROID)
    # 创建共享库版本
    add_library(mnist_inference_lib SHARED ../inference/android_inference.cpp)
    target_include_directories(mnist_inference_lib PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
    target_link_libraries(mnist_inference_lib ${ONNXRUNTIME_LIB} log)
    
    # 设置输出目录
    set_target_properties(mnist_inference_lib PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../android_libs/${ANDROID_ABI}"
    )
endif()

# 编译优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(ANDROID AND CMAKE_ANDROID_ARCH_ABI MATCHES "arm")
        target_compile_options(mnist_inference_cpp PRIVATE -mfpu=neon -O3)
        if(TARGET mnist_inference_lib)
            target_compile_options(mnist_inference_lib PRIVATE -mfpu=neon -O3)
        endif()
    endif()
endif()
```

## 8. Android NDK编译

### 8.1 Android专用推理代码

```cpp
// inference/android_inference.cpp
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

#define LOG_TAG "MNISTInference"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

class AndroidMNISTInference {
private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memory_info;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int64_t> input_shape;

public:
    AndroidMNISTInference() {
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AndroidMNISTInference");
        memory_info = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
        );
    }
    
    bool loadModel(const std::string& model_path) {
        try {
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            
            session = std::make_unique<Ort::Session>(*env, model_path.c_str(), session_options);
            
            // 获取输入输出信息
            size_t num_input_nodes = session->GetInputCount();
            for (size_t i = 0; i < num_input_nodes; i++) {
                auto input_name = session->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions{});
                input_names.push_back(input_name.release());
                
                Ort::TypeInfo input_type_info = session->GetInputTypeInfo(i);
                auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
                input_shape = input_tensor_info.GetShape();
            }
            
            size_t num_output_nodes = session->GetOutputCount();
            for (size_t i = 0; i < num_output_nodes; i++) {
                auto output_name = session->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions{});
                output_names.push_back(output_name.release());
            }
            
            LOGI("模型加载成功");
            return true;
        } catch (const std::exception& e) {
            LOGE("模型加载失败: %s", e.what());
            return false;
        }
    }
    
    std::pair<int, float> inference(const std::vector<float>& input_data) {
        try {
            // 预处理
            std::vector<float> processed_data = input_data;
            const float mean = 0.1307f;
            const float std = 0.3081f;
            
            for (auto& pixel : processed_data) {
                pixel = (pixel - mean) / std;
            }
            
            // 创建输入tensor
            auto input_tensor = Ort::Value::CreateTensor<float>(
                *memory_info,
                processed_data.data(),
                processed_data.size(),
                input_shape.data(),
                input_shape.size()
            );
            
            // 运行推理
            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_names.data(),
                &input_tensor,
                1,
                output_names.data(),
                1
            );
            
            // 获取输出
            float* output_data = output_tensors.front().GetTensorMutableData<float>();
            size_t output_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
            
            // 应用softmax
            std::vector<float> probabilities(output_size);
            float max_logit = *std::max_element(output_data, output_data + output_size);
            float sum = 0.0f;
            
            for (size_t i = 0; i < output_size; ++i) {
                probabilities[i] = std::exp(output_data[i] - max_logit);
                sum += probabilities[i];
            }
            
            for (auto& prob : probabilities) {
                prob /= sum;
            }
            
            // 找到预测类别
            auto max_it = std::max_element(probabilities.begin(), probabilities.end());
            int predicted_class = std::distance(probabilities.begin(), max_it);
            float confidence = *max_it;
            
            return {predicted_class, confidence};
            
        } catch (const std::exception& e) {
            LOGE("推理失败: %s", e.what());
            return {-1, 0.0f};
        }
    }
};

// 全局推理引擎实例
static std::unique_ptr<AndroidMNISTInference> g_inference_engine;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_example_mnistinference_MNISTInference_loadModel(JNIEnv *env, jobject thiz, jstring model_path) {
    const char* path = env->GetStringUTFChars(model_path, nullptr);
    
    g_inference_engine = std::make_unique<AndroidMNISTInference>();
    bool success = g_inference_engine->loadModel(std::string(path));
    
    env->ReleaseStringUTFChars(model_path, path);
    return success;
}

JNIEXPORT jfloatArray JNICALL
Java_com_example_mnistinference_MNISTInference_inference(JNIEnv *env, jobject thiz, jfloatArray input_data) {
    if (!g_inference_engine) {
        LOGE("推理引擎未初始化");
        return nullptr;
    }
    
    // 获取输入数据
    jsize input_size = env->GetArrayLength(input_data);
    jfloat* input_ptr = env->GetFloatArrayElements(input_data, nullptr);
    
    std::vector<float> input_vector(input_ptr, input_ptr + input_size);
    
    // 执行推理
    auto [predicted_class, confidence] = g_inference_engine->inference(input_vector);
    
    // 创建返回数组 [predicted_class, confidence]
    jfloatArray result = env->NewFloatArray(2);
    jfloat result_data[2] = {static_cast<jfloat>(predicted_class), confidence};
    env->SetFloatArrayRegion(result, 0, 2, result_data);
    
    env->ReleaseFloatArrayElements(input_data, input_ptr, JNI_ABORT);
    
    return result;
}

} // extern "C"
```

### 8.2 编译脚本

```bash
#!/bin/bash
# build/build_android.sh

set -e

# 设置变量
ANDROID_NDK_HOME=${ANDROID_NDK_HOME:-"/opt/homebrew/share/android-ndk"}
ANDROID_API=21
PROJECT_DIR=$(pwd)/..

# 下载ONNX Runtime for Android
download_onnxruntime() {
    local abi=$1
    local version="1.16.0"
    local url="https://github.com/microsoft/onnxruntime/releases/download/v${version}/onnxruntime-android-${abi}-${version}.tgz"
    local target_dir="onnxruntime-android-${abi}"
    
    if [ ! -d "${target_dir}" ]; then
        echo "下载 ONNX Runtime for ${abi}..."
        wget -O "${target_dir}.tgz" "${url}"
        tar -xzf "${target_dir}.tgz"
        rm "${target_dir}.tgz"
    fi
}

# 编译函数
build_for_abi() {
    local abi=$1
    echo "编译 ${abi} 版本..."
    
    # 下载对应的ONNX Runtime
    download_onnxruntime ${abi}
    
    # 创建构建目录
    mkdir -p "build_${abi}"
    cd "build_${abi}"
    
    # 配置CMake
    cmake \
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=${abi} \
        -DANDROID_NATIVE_API_LEVEL=${ANDROID_API} \
        -DCMAKE_BUILD_TYPE=Release \
        -DANDROID_STL=c++_shared \
        ..
    
    # 编译
    make -j$(nproc)
    
    cd ..
}

# 主函数
main() {
    echo "开始编译Android版本..."
    
    # 创建输出目录
    mkdir -p "${PROJECT_DIR}/android_libs"
    
    # 编译不同架构
    for abi in "arm64-v8a" "armeabi-v7a" "x86" "x86_64"; do
        build_for_abi ${abi}
    done
    
    echo "编译完成！"
    echo "库文件位置: ${PROJECT_DIR}/android_libs/"
}

# 执行主函数
main
```

### 8.3 可执行文件编译

```bash
#!/bin/bash
# build/build_executable.sh

# 编译macOS版本
build_macos() {
    echo "编译macOS版本..."
    
    # 下载ONNX Runtime for macOS
    if [ ! -d "onnxruntime-osx-x86_64-1.16.0" ]; then
        wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-osx-x86_64-1.16.0.tgz
        tar -xzf onnxruntime-osx-x86_64-1.16.0.tgz
        rm onnxruntime-osx-x86_64-1.16.0.tgz
    fi
    
    mkdir -p build_macos
    cd build_macos
    
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j$(sysctl -n hw.ncpu)
    
    cd ..
}

# 编译Android可执行文件
build_android_executable() {
    local abi=$1
    echo "编译Android可执行文件 (${abi})..."
    
    mkdir -p "build_android_exe_${abi}"
    cd "build_android_exe_${abi}"
    
    # 创建简单的CMakeLists用于可执行文件
    cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.16)
project(MNISTInferenceExecutable)

set(CMAKE_CXX_STANDARD 17)

set(ONNXRUNTIME_ROOT_PATH "\${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-android-${abi}")
set(ONNXRUNTIME_INCLUDE_DIRS \${ONNXRUNTIME_ROOT_PATH}/include)
set(ONNXRUNTIME_LIB \${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so)

add_executable(mnist_inference ../../inference/cpp_inference.cpp)
target_include_directories(mnist_inference PRIVATE \${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(mnist_inference \${ONNXRUNTIME_LIB})
EOF
    
    cmake \
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=${abi} \
        -DANDROID_NATIVE_API_LEVEL=21 \
        -DCMAKE_BUILD_TYPE=Release \
        -DANDROID_STL=c++_shared \
        .
    
    make -j$(nproc)
    
    # 复制可执行文件到输出目录
    mkdir -p "../android_executables/${abi}"
    cp mnist_inference "../android_executables/${abi}/"
    cp "../onnxruntime-android-${abi}/lib/libonnxruntime.so" "../android_executables/${abi}/"
    
    cd ..
}

# 主函数
main() {
    # 编译macOS版本
    build_macos
    
    # 编译Android可执行文件
    mkdir -p android_executables
    for abi in "arm64-v8a" "armeabi-v7a"; do
        build_android_executable ${abi}
    done
    
    echo "所有版本编译完成！"
}

main
```

## 9. 手机部署与测试

### 9.1 部署脚本

```bash
#!/bin/bash
# deploy_and_test.sh

set -e

PROJECT_DIR=$(pwd)
DEVICE_DIR="/data/local/tmp/mnist_inference"

# 检查设备连接
check_device() {
    if ! adb devices | grep -q "device$"; then
        echo "错误: 没有检测到Android设备"
        echo "请确保设备已连接并开启USB调试"
        exit 1
    fi
    echo "设备连接正常"
}

# 获取设备架构
get_device_arch() {
    local arch=$(adb shell getprop ro.product.cpu.abi | tr -d '\r')
    echo $arch
}

# 部署文件到设备
deploy_to_device() {
    local arch=$1
    
    echo "部署到设备 (${arch})..."
    
    # 创建设备目录
    adb shell "mkdir -p ${DEVICE_DIR}"
    
    # 推送可执行文件和库
    adb push "android_executables/${arch}/mnist_inference" "${DEVICE_DIR}/"
    adb push "android_executables/${arch}/libonnxruntime.so" "${DEVICE_DIR}/"
    adb push "models/mnist_model.onnx" "${DEVICE_DIR}/"
    
    # 设置执行权限
    adb shell "chmod +x ${DEVICE_DIR}/mnist_inference"
    
    echo "文件部署完成"
}

# 生成测试数据
generate_test_data() {
    cat > generate_test_data.py << 'EOF'
import numpy as np
import json
import torchvision
import torchvision.transforms as transforms

# 加载MNIST测试数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=False, transform=transform
)

# 生成测试样本
test_samples = []
for i in range(10):
    image, label = test_dataset[i]
    # 转换为原始像素值 (0-1范围，未归一化)
    raw_image = (image * 0.3081 + 0.1307).clamp(0, 1)
    test_samples.append({
        'id': i,
        'label': int(label),
        'data': raw_image.flatten().tolist()
    })

# 保存测试数据
with open('test_data.json', 'w') as f:
    json.dump(test_samples, f, indent=2)

print("测试数据已生成: test_data.json")
EOF

    python generate_test_data.py
    adb push test_data.json "${DEVICE_DIR}/"
}

# 在设备上运行测试
run_device_test() {
    echo "在设备上运行推理测试..."
    
    # 设置库路径并运行
    adb shell "cd ${DEVICE_DIR} && LD_LIBRARY_PATH=. ./mnist_inference > device_results.txt 2>&1"
    
    # 获取结果
    adb pull "${DEVICE_DIR}/device_results.txt" "./results/"
    
    echo "设备测试完成，结果保存到 ./results/device_results.txt"
}

# 结果对比
compare_results() {
    echo "=== 结果对比分析 ==="
    
    cat > compare_results.py << 'EOF'
import json
import numpy as np

def load_results(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None

def parse_device_results(file_path):
    """解析设备输出的文本结果"""
    results = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "样本" in line and "预测=" in line:
                    # 解析行: "样本 0: 真实=7, 预测=7, 置信度=0.9999, 时间=1.23ms"
                    parts = line.strip().split(', ')
                    sample_info = parts[0].split(': ')
                    sample_id = int(sample_info[0].split(' ')[1])
                    true_label = int(sample_info[1].split('=')[1])
                    predicted = int(parts[1].split('=')[1])
                    confidence = float(parts[2].split('=')[1])
                    time_ms = float(parts[3].split('=')[1].replace('ms', ''))
                    
                    results.append({
                        'sample_id': sample_id,
                        'true_label': true_label,
                        'predicted_class': predicted,
                        'confidence': confidence,
                        'inference_time_ms': time_ms
                    })
    except Exception as e:
        print(f"解析设备结果失败: {e}")
    
    return results

# 加载结果
python_results = load_results('./results/python_inference_results.json')
cpp_results = load_results('./results/cpp_inference_results.json')
device_results = parse_device_results('./results/device_results.txt')

print("=== 推理结果对比 ===")
print(f"Python ONNX: {len(python_results['results']) if python_results else 0} 个样本")
print(f"C++ ONNX:    {len(cpp_results) if cpp_results else 0} 个样本") 
print(f"设备推理:    {len(device_results)} 个样本")

if python_results and device_results:
    print("\n=== 准确率对比 ===")
    
    # Python结果
    py_correct = sum(1 for r in python_results['results'] if r['predicted_class'] == r['true_label'])
    py_accuracy = py_correct / len(python_results['results'])
    py_avg_time = python_results['summary']['average_inference_time_ms']
    
    # 设备结果
    dev_correct = sum(1 for r in device_results if r['predicted_class'] == r['true_label'])
    dev_accuracy = dev_correct / len(device_results) if device_results else 0
    dev_avg_time = np.mean([r['inference_time_ms'] for r in device_results]) if device_results else 0
    
    print(f"Python (macOS): {py_accuracy:.2%}, 平均时间: {py_avg_time:.2f}ms")
    print(f"设备 (Android): {dev_accuracy:.2%}, 平均时间: {dev_avg_time:.2f}ms")
    
    print("\n=== 逐样本对比 ===")
    for i in range(min(len(python_results['results']), len(device_results))):
        py_r = python_results['results'][i]
        dev_r = device_results[i]
        
        match = "✓" if py_r['predicted_class'] == dev_r['predicted_class'] else "✗"
        print(f"样本{i}: macOS={py_r['predicted_class']}({py_r['confidence']:.3f}), "
              f"Android={dev_r['predicted_class']}({dev_r['confidence']:.3f}) {match}")

print("\n=== 性能对比 ===")
if python_results and device_results:
    speedup = py_avg_time / dev_avg_time if dev_avg_time > 0 else 0
    print(f"速度提升: {speedup:.2f}x" if speedup > 1 else f"速度下降: {1/speedup:.2f}x")

EOF

    python compare_results.py
}

# 主函数
main() {
    echo "=== MNIST模型手机部署测试 ==="
    
    # 检查设备
    check_device
    
    # 获取设备架构
    DEVICE_ARCH=$(get_device_arch)
    echo "设备架构: ${DEVICE_ARCH}"
    
    # 检查是否有对应架构的编译文件
    if [ ! -f "android_executables/${DEVICE_ARCH}/mnist_inference" ]; then
        echo "错误: 没有找到 ${DEVICE_ARCH} 架构的编译文件"
        echo "请先运行编译脚本: ./build/build_executable.sh"
        exit 1
    fi
    
    # 生成测试数据
    generate_test_data
    
    # 部署到设备
    deploy_to_device ${DEVICE_ARCH}
    
    # 运行设备测试
    run_device_test
    
    # 对比结果
    compare_results
    
    echo "=== 部署测试完成 ==="
}

# 执行主函数
main
```

## 10. 完整运行脚本

### 10.1 一键运行脚本

```bash
#!/bin/bash
# run_complete_pipeline.sh

set -e

echo "=== MNIST完整部署流程 ==="
echo "开始时间: $(date)"

# 创建项目结构
setup_project() {
    echo "1. 设置项目结构..."
    mkdir -p {train,inference,build,models,data,results,android_libs,android_executables}
}

# 训练模型
train_model() {
    echo "2. 训练MNIST模型..."
    cd train
    python train_model.py
    cd ..
}

# 量化模型
quantize_model() {
    echo "3. 量化模型..."
    cd train
    python quantize_model.py
    cd ..
}

# 导出ONNX
export_onnx() {
    echo "4. 导出ONNX模型..."
    cd train
    python export_onnx.py
    cd ..
}

# 可视化模型
visualize_model() {
    echo "5. 启动Netron可视化 (后台运行)..."
    python -c "
import netron
import threading
import time

def start_netron():
    netron.start('./models/mnist_model.onnx', port=8080, browse=True)

thread = threading.Thread(target=start_netron, daemon=True)
thread.start()
print('Netron已启动: http://localhost:8080')
time.sleep(2)
" &
}

# Python推理
python_inference() {
    echo "6. Python ONNX推理..."
    cd inference
    python python_inference.py
    cd ..
}

# 编译C++版本
build_cpp() {
    echo "7. 编译C++版本..."
    cd build
    
    # 下载ONNX Runtime for macOS
    if [ ! -d "onnxruntime-osx-x86_64-1.16.0" ]; then
        echo "下载ONNX Runtime for macOS..."
        wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-osx-x86_64-1.16.0.tgz
        tar -xzf onnxruntime-osx-x86_64-1.16.0.tgz
        rm onnxruntime-osx-x86_64-1.16.0.tgz
    fi
    
    mkdir -p build_macos
    cd build_macos
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j$(sysctl -n hw.ncpu)
    cd ../..
}

# C++推理
cpp_inference() {
    echo "8. C++推理测试..."
    cd build/build_macos
    ./mnist_inference_cpp
    cd ../..
}

# 编译Android版本
build_android() {
    echo "9. 编译Android版本..."
    cd build
    chmod +x build_android.sh build_executable.sh
    ./build_executable.sh
    cd ..
}

# 部署到手机
deploy_to_phone() {
    echo "10. 部署到手机并测试..."
    chmod +x deploy_and_test.sh
    ./deploy_and_test.sh
}

# 主函数
main() {
    # 检查依赖
    echo "检查依赖..."
    python -c "import torch, torchvision, onnx, onnxruntime, netron; print('Python依赖检查通过')"
    
    if [ -z "$ANDROID_NDK_HOME" ]; then
        echo "警告: ANDROID_NDK_HOME未设置，Android编译将被跳过"
        SKIP_ANDROID=true
    fi
    
    # 执行流程
    setup_project
    train_model
    quantize_model
    export_onnx
    visualize_model
    python_inference
    build_cpp
    cpp_inference
    
    if [ "$SKIP_ANDROID" != "true" ]; then
        build_android
        
        # 检查是否有Android设备
        if adb devices | grep -q "device$"; then
            deploy_to_phone
        else
            echo "没有检测到Android设备，跳过手机部署"
        fi
    fi
    
    echo "=== 完整流程执行完成 ==="
    echo "结束时间: $(date)"
    
    # 显示结果总结
    echo ""
    echo "=== 结果文件 ==="
    echo "模型文件:"
    ls -la models/
    echo ""
    echo "推理结果:"
    ls -la results/
    echo ""
    echo "Android库文件:"
    if [ -d "android_libs" ]; then
        find android_libs -name "*.so" -o -name "*.a"
    fi
    echo ""
    echo "Android可执行文件:"
    if [ -d "android_executables" ]; then
        find android_executables -name "mnist_inference"
    fi
}

# 执行主函数
main "$@"
```

### 10.2 使用说明

```bash
# 1. 克隆或创建项目目录
mkdir mnist_complete_project
cd mnist_complete_project

# 2. 复制所有脚本文件到对应目录

# 3. 设置Android NDK (如果需要Android编译)
export ANDROID_NDK_HOME=/path/to/android-ndk

# 4. 连接Android设备 (如果需要手机测试)
adb devices

# 5. 运行完整流程
chmod +x run_complete_pipeline.sh
./run_complete_pipeline.sh
```

这个完整的流程包括：

1. ✅ **PyTorch训练** - 训练MNIST分类模型
2. ✅ **模型量化** - 使用PyTorch量化API
3. ✅ **ONNX导出** - 转换为ONNX格式
4. ✅ **Netron可视化** - 查看模型结构
5. ✅ **Python推理** - ONNX Runtime Python API
6. ✅ **C++推理** - ONNX Runtime C++ API
7. ✅ **NDK编译** - 编译ARM库文件和可执行文件
8. ✅ **手机部署** - adb部署并运行测试
9. ✅ **结果对比** - 对比macOS和Android推理结果

整个流程实现了从模型训练到移动端部署的完整自动化，并提供了详细的性能和准确率对比分析。
