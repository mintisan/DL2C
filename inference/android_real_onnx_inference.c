#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "onnxruntime_c_api.h"

// 推理上下文结构体
typedef struct {
    const OrtApi* ort_api;
    OrtEnv* env;
    OrtSession* session;
    OrtMemoryInfo* memory_info;
    OrtAllocator* allocator;
    char** input_names;
    char** output_names;
    size_t num_inputs;
    size_t num_outputs;
} InferenceContext;

// 推理结果结构体
typedef struct {
    int sample_id;
    int original_mnist_index;
    int true_label;
    int predicted_class;
    float confidence;
    float* probabilities;
    size_t prob_count;
    double inference_time_ms;
    int is_correct;
} InferenceResult;

// 标签映射结构体
typedef struct {
    int* indices;
    int* labels;
    int num_samples;
} LabelMap;

// 全局ORT API指针
const OrtApi* g_ort = NULL;

// 错误处理宏
#define CHECK_STATUS(status) \
    if (status != NULL) { \
        const char* msg = g_ort->GetErrorMessage(status); \
        printf("错误: %s\n", msg); \
        g_ort->ReleaseStatus(status); \
        return -1; \
    }

// 预处理函数（与Android C++版本保持一致）
void preprocess_image(float* data, size_t data_size) {
    // MNIST标准化参数（与macOS C++和Python版本保持一致）
    const float mean = 0.1307f;
    const float std = 0.3081f;
    
    // 标准化: (pixel - mean) / std
    for (size_t i = 0; i < data_size; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

// Softmax函数（与Android C++版本保持一致）
void softmax(const float* input, float* output, size_t size) {
    // 数值稳定性：减去最大值
    float max_val = input[0];
    for (size_t i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // 计算exp和sum
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    // 归一化
    for (size_t i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// 动态加载所有样本的标签信息（与Android C++版本保持一致）
int load_labels_from_metadata(LabelMap* label_map) {
    FILE* metadata_file = fopen("test_data_mnist/metadata.json", "r");
    if (!metadata_file) {
        printf("❌ 无法打开元数据文件: test_data_mnist/metadata.json\n");
        printf("使用默认标签映射...\n");
        
        // 提供前10个样本的后备标签
        label_map->num_samples = 10;
        label_map->indices = (int*)malloc(10 * sizeof(int));
        label_map->labels = (int*)malloc(10 * sizeof(int));
        
        int default_labels[] = {2, 1, 1, 1, 2, 6, 3, 8, 2, 6};
        for (int i = 0; i < 10; i++) {
            label_map->indices[i] = i;
            label_map->labels[i] = default_labels[i];
        }
        return 0;
    }
    
    // 简单解析JSON获取标签信息
    char line[1024];
    int* labels = (int*)malloc(1000 * sizeof(int));  // 预分配空间
    int num_samples = 0;
    int label_count = 0;
    
    while (fgets(line, sizeof(line), metadata_file)) {
        // 获取样本数量
        char* pos = strstr(line, "\"num_samples\":");
        if (pos) {
            pos += 14;
            while (*pos == ' ' || *pos == '"') pos++;
            char* end_pos = strchr(pos, ',');
            if (!end_pos) end_pos = strchr(pos, '}');
            if (end_pos) {
                *end_pos = '\0';
                num_samples = atoi(pos);
            }
        }
        
        // 获取标签
        pos = strstr(line, "\"true_label\":");
        if (pos) {
            pos += 13;
            while (*pos == ' ' || *pos == '"') pos++;
            char* end_pos = strchr(pos, ',');
            if (end_pos) {
                *end_pos = '\0';
                labels[label_count++] = atoi(pos);
            }
        }
    }
    fclose(metadata_file);
    
    // 构建标签映射
    label_map->num_samples = (num_samples < label_count) ? num_samples : label_count;
    label_map->indices = (int*)malloc(label_map->num_samples * sizeof(int));
    label_map->labels = (int*)malloc(label_map->num_samples * sizeof(int));
    
    for (int i = 0; i < label_map->num_samples; i++) {
        label_map->indices[i] = i;
        label_map->labels[i] = labels[i];
    }
    
    free(labels);
    printf("✓ 已从metadata.json加载 %d 个样本的标签信息\n", label_map->num_samples);
    return 0;
}

// 释放标签映射
void free_label_map(LabelMap* label_map) {
    if (label_map->indices) {
        free(label_map->indices);
        label_map->indices = NULL;
    }
    if (label_map->labels) {
        free(label_map->labels);
        label_map->labels = NULL;
    }
    label_map->num_samples = 0;
}

// 加载测试数据
float* load_test_data(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("警告: 无法打开测试数据文件: %s\n", filename);
        printf("使用随机测试数据...\n");
        
        // 生成随机测试数据
        float* data = (float*)malloc(784 * sizeof(float));
        for (int i = 0; i < 784; i++) {
            data[i] = (float)rand() / (float)RAND_MAX;
        }
        return data;
    }
    
    // 只读取图像数据（784个float值）
    float* data = (float*)malloc(784 * sizeof(float));
    size_t read_count = fread(data, sizeof(float), 784, file);
    fclose(file);
    
    if (read_count != 784) {
        printf("警告: 读取数据不完整，期望784个float，实际读取%zu个\n", read_count);
    }
    
    printf("✓ 加载测试数据: %s\n", filename);
    return data;
}

// 初始化推理上下文
int init_inference_context(InferenceContext* ctx, const char* model_path) {
    printf("=== Android 真实 ONNX Runtime C API 推理测试 ===\n");
    printf("使用真正的 ONNX Runtime Android 版本\n");
    printf("初始化ONNX Runtime C API推理引擎...\n");
    
    // 获取ORT API
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ctx->ort_api = g_ort;
    
    // 创建环境
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "AndroidCInference", &ctx->env);
    CHECK_STATUS(status);
    
    // 创建会话选项
    OrtSessionOptions* session_options;
    status = g_ort->CreateSessionOptions(&session_options);
    CHECK_STATUS(status);
    
    // 设置线程数
    status = g_ort->SetIntraOpNumThreads(session_options, 1);
    if (status != NULL) g_ort->ReleaseStatus(status);
    
    // 创建会话
    status = g_ort->CreateSession(ctx->env, model_path, session_options, &ctx->session);
    if (status != NULL) {
        printf("错误: 加载模型失败: %s\n", model_path);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseStatus(status);
        return -1;
    }
    
    // 释放会话选项
    g_ort->ReleaseSessionOptions(session_options);
    
    // 创建内存信息
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ctx->memory_info);
    CHECK_STATUS(status);
    
    // 获取默认分配器
    status = g_ort->GetAllocatorWithDefaultOptions(&ctx->allocator);
    CHECK_STATUS(status);
    
    // 获取输入信息
    status = g_ort->SessionGetInputCount(ctx->session, &ctx->num_inputs);
    CHECK_STATUS(status);
    
    ctx->input_names = (char**)malloc(ctx->num_inputs * sizeof(char*));
    for (size_t i = 0; i < ctx->num_inputs; i++) {
        status = g_ort->SessionGetInputName(ctx->session, i, ctx->allocator, &ctx->input_names[i]);
        CHECK_STATUS(status);
    }
    
    // 获取输出信息
    status = g_ort->SessionGetOutputCount(ctx->session, &ctx->num_outputs);
    CHECK_STATUS(status);
    
    ctx->output_names = (char**)malloc(ctx->num_outputs * sizeof(char*));
    for (size_t i = 0; i < ctx->num_outputs; i++) {
        status = g_ort->SessionGetOutputName(ctx->session, i, ctx->allocator, &ctx->output_names[i]);
        CHECK_STATUS(status);
    }
    
    printf("✓ ONNX Runtime 初始化成功\n");
    printf("✓ 模型加载成功: %s\n", model_path);
    
    return 0;
}

// 执行推理
int run_inference(InferenceContext* ctx, float* input_data, size_t input_size, 
                 int sample_id, int original_idx, int true_label, InferenceResult* result) {
    clock_t start_time = clock();
    
    // ✅ 预处理（与Android C++版本保持一致）
    preprocess_image(input_data, input_size);
    
    // 创建输入tensor
    int64_t input_shape[] = {1, 1, 28, 28};
    OrtValue* input_tensor = NULL;
    
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(
        ctx->memory_info,
        input_data,
        input_size * sizeof(float),
        input_shape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    );
    
    if (status != NULL) {
        g_ort->ReleaseStatus(status);
        return -1;
    }
    
    // 准备输入输出数组
    const OrtValue* inputs[] = { input_tensor };
    OrtValue* outputs[1] = { NULL };
    
    // 运行推理
    status = g_ort->Run(
        ctx->session,
        NULL,
        (const char* const*)ctx->input_names,
        inputs,
        ctx->num_inputs,
        (const char* const*)ctx->output_names,
        ctx->num_outputs,
        outputs
    );
    
    if (status != NULL) {
        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseStatus(status);
        return -1;
    }
    
    // 获取输出数据
    float* output_data;
    status = g_ort->GetTensorMutableData(outputs[0], (void**)&output_data);
    if (status != NULL) {
        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseValue(outputs[0]);
        g_ort->ReleaseStatus(status);
        return -1;
    }
    
    // ✅ 复制输出数据并应用softmax（与Android C++版本保持一致）
    float logits[10];
    for (int i = 0; i < 10; i++) {
        logits[i] = output_data[i];
    }
    
    result->prob_count = 10;
    result->probabilities = (float*)malloc(10 * sizeof(float));
    softmax(logits, result->probabilities, 10);
    
    // 找到预测类别
    result->predicted_class = 0;
    result->confidence = result->probabilities[0];
    for (int i = 1; i < 10; i++) {
        if (result->probabilities[i] > result->confidence) {
            result->confidence = result->probabilities[i];
            result->predicted_class = i;
        }
    }
    
    // 计算推理时间
    clock_t end_time = clock();
    result->inference_time_ms = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    // 设置结果信息
    result->sample_id = sample_id;
    result->original_mnist_index = original_idx;
    result->true_label = true_label;
    result->is_correct = (result->predicted_class == true_label);
    
    // 释放资源
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(outputs[0]);
    
    return 0;
}

// 清理推理上下文
void cleanup_inference_context(InferenceContext* ctx) {
    if (ctx->session) {
        g_ort->ReleaseSession(ctx->session);
        ctx->session = NULL;
    }
    
    if (ctx->memory_info) {
        g_ort->ReleaseMemoryInfo(ctx->memory_info);
        ctx->memory_info = NULL;
    }
    
    if (ctx->env) {
        g_ort->ReleaseEnv(ctx->env);
        ctx->env = NULL;
    }
    
    // 释放输入输出名称
    if (ctx->input_names && ctx->allocator) {
        for (size_t i = 0; i < ctx->num_inputs; i++) {
            if (ctx->input_names[i]) {
                // 忽略AllocatorFree的返回值，因为在清理阶段错误处理不关键
                (void)ctx->ort_api->AllocatorFree(ctx->allocator, ctx->input_names[i]);
            }
        }
        free(ctx->input_names);
        ctx->input_names = NULL;
    }
    
    if (ctx->output_names && ctx->allocator) {
        for (size_t i = 0; i < ctx->num_outputs; i++) {
            if (ctx->output_names[i]) {
                // 忽略AllocatorFree的返回值，因为在清理阶段错误处理不关键
                (void)ctx->ort_api->AllocatorFree(ctx->allocator, ctx->output_names[i]);
            }
        }
        free(ctx->output_names);
        ctx->output_names = NULL;
    }
}

// 释放推理结果
void free_inference_result(InferenceResult* result) {
    if (result->probabilities) {
        free(result->probabilities);
        result->probabilities = NULL;
    }
}

// 保存结果到文件
void save_results_to_file(InferenceResult* results, int num_results, double avg_time, double accuracy) {
    FILE* file = fopen("results/android_real_onnx_c_results.txt", "w");
    if (!file) {
        printf("无法创建结果文件\n");
        return;
    }
    
    fprintf(file, "Android 真实 ONNX Runtime C API 推理结果\n");
    fprintf(file, "==========================================\n");
    fprintf(file, "测试样本数: %d\n", num_results);
    fprintf(file, "准确率: %.2f%%\n", accuracy);
    fprintf(file, "平均推理时间: %.2f ms\n", avg_time);
    fprintf(file, "推理 FPS: %.0f\n", 1000.0 / avg_time);
    fprintf(file, "使用框架: ONNX Runtime C API (Android)\n");
    fclose(file);
    
    printf("✓ 结果已保存到 results/android_real_onnx_c_results.txt\n");
}

// 主函数
int main() {
    printf("启动 Android 真实 ONNX Runtime C API MNIST 推理程序...\n");
    
    InferenceContext ctx = {0};
    const char* model_path = "models/mnist_model.onnx";
    
    // 初始化推理上下文
    if (init_inference_context(&ctx, model_path) != 0) {
        printf("初始化失败\n");
        return -1;
    }
    
    printf("\n=== 开始 Android 真实 ONNX Runtime C API 推理测试 ===\n");
    
    // 加载标签信息
    LabelMap label_map = {0};
    if (load_labels_from_metadata(&label_map) != 0) {
        printf("加载标签信息失败\n");
        cleanup_inference_context(&ctx);
        return -1;
    }
    
    // 测试所有样本（与Android C++版本完全一致）
    int num_samples = label_map.num_samples;
    printf("开始推理 %d 个样本...\n", num_samples);
    
    InferenceResult* results = (InferenceResult*)malloc(num_samples * sizeof(InferenceResult));
    double total_time = 0.0;
    int correct_predictions = 0;
    
    for (int idx = 0; idx < num_samples; idx++) {
        // 构建文件名
        char filename[256];
        snprintf(filename, sizeof(filename), "test_data_mnist/image_%03d.bin", idx);
        
        // 加载测试数据
        float* input_data = load_test_data(filename);
        if (!input_data) {
            printf("加载样本 %d 失败\n", idx);
            continue;
        }
        
        // 获取真实标签
        int true_label = label_map.labels[idx];
        
        // 执行推理
        if (run_inference(&ctx, input_data, 784, idx, idx, true_label, &results[idx]) == 0) {
            total_time += results[idx].inference_time_ms;
            if (results[idx].is_correct) {
                correct_predictions++;
            }
            
            // 显示进度（每10个样本，与Android C++版本保持一致）
            if ((idx + 1) % 10 == 0) {
                double current_accuracy = (double)correct_predictions / (idx + 1) * 100;
                printf("完成 %3d/%d 样本，当前准确率: %.1f%%\n", 
                       idx + 1, num_samples, current_accuracy);
            }
        } else {
            printf("样本 %d 推理失败\n", idx);
        }
        
        free(input_data);
    }
    
    // 计算统计信息
    double avg_time = total_time / num_samples;
    double accuracy = (double)correct_predictions / num_samples * 100.0;
    int wrong_count = num_samples - correct_predictions;
    
    printf("\n=== Android 真实 ONNX Runtime C API 推理结果统计 ===\n");
    printf("总样本数: %d\n", num_samples);
    printf("正确预测: %d\n", correct_predictions);
    printf("准确率: %.2f%%\n", accuracy);
    printf("平均推理时间: %.2f ms\n", avg_time);
    printf("推理速度: %.1f FPS\n", 1000.0 / avg_time);
    
    // 显示错误样本（与Android C++版本保持一致）
    if (wrong_count > 0) {
        printf("\n❌ 错误预测样本 (%d 个):\n", wrong_count);
        int shown = 0;
        for (int idx = 0; idx < num_samples && shown < 5; idx++) {
            if (!results[idx].is_correct) {
                printf("  样本 %3d: 真实=%d, 预测=%d, 时间=%.3fms\n",
                       idx, results[idx].true_label, results[idx].predicted_class, 
                       results[idx].inference_time_ms);
                shown++;
            }
        }
        if (wrong_count > 5) {
            printf("  ... 还有 %d 个错误样本\n", wrong_count - 5);
        }
    }
    
    // 保存结果
    save_results_to_file(results, num_samples, avg_time, accuracy);
    
    // 清理资源
    for (int i = 0; i < num_samples; i++) {
        free_inference_result(&results[i]);
    }
    free(results);
    free_label_map(&label_map);
    cleanup_inference_context(&ctx);
    
    printf("\nAndroid 真实 ONNX Runtime C API 推理测试完成！\n");
    
    return 0;
} 