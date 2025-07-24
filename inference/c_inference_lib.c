#include "c_inference_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "onnxruntime_c_api.h"

// 推理上下文结构体（完整定义）
typedef struct InferenceContext {
    const OrtApi* ort_api;
    OrtEnv* env;
    OrtSession* session;
    OrtMemoryInfo* memory_info;
    OrtAllocator* allocator;
    char** input_names;
    char** output_names;
    size_t num_inputs;
    size_t num_outputs;
    char* model_path;
} InferenceContext;

// 全局ORT API指针
static const OrtApi* g_ort = NULL;

// 错误处理宏
#define CHECK_STATUS_RETURN(status, retval) \
    if (status != NULL) { \
        const char* msg = g_ort->GetErrorMessage(status); \
        printf("错误: %s\n", msg); \
        g_ort->ReleaseStatus(status); \
        return retval; \
    }

// === 内部工具函数 ===

// 简单的JSON解析函数
static int parse_json_int(const char* line, const char* key) {
    char* pos = strstr(line, key);
    if (!pos) return -1;
    
    pos += strlen(key);
    while (*pos == ' ' || *pos == ':' || *pos == '"') pos++;
    
    char* end_pos = pos;
    while (*end_pos && *end_pos != ',' && *end_pos != '}' && *end_pos != '"') end_pos++;
    
    char value_str[32];
    size_t len = end_pos - pos;
    if (len >= sizeof(value_str)) len = sizeof(value_str) - 1;
    
    strncpy(value_str, pos, len);
    value_str[len] = '\0';
    
    return atoi(value_str);
}

// 预处理函数（MNIST标准化）
static void preprocess_image(float* input_data, size_t data_size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    
    for (size_t i = 0; i < data_size; i++) {
        input_data[i] = (input_data[i] - mean) / std;
    }
}

// Softmax函数
static void softmax(float* input, float* output, size_t size) {
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

// === 公开API实现 ===

InferenceHandle inference_create(const char* model_path) {
    if (!model_path) {
        printf("错误: 模型路径为空\n");
        return NULL;
    }
    
    printf("初始化ONNX Runtime C API推理引擎...\n");
    
    // 分配推理上下文
    InferenceContext* ctx = (InferenceContext*)calloc(1, sizeof(InferenceContext));
    if (!ctx) {
        printf("错误: 内存分配失败\n");
        return NULL;
    }
    
    // 复制模型路径
    ctx->model_path = (char*)malloc(strlen(model_path) + 1);
    strcpy(ctx->model_path, model_path);
    
    // 获取ORT API
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ctx->ort_api = g_ort;
    
    // 创建环境
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "CInferenceLib", &ctx->env);
    CHECK_STATUS_RETURN(status, NULL);
    
    // 创建会话选项
    OrtSessionOptions* session_options;
    status = g_ort->CreateSessionOptions(&session_options);
    CHECK_STATUS_RETURN(status, NULL);
    
    status = g_ort->SetIntraOpNumThreads(session_options, 1);
    CHECK_STATUS_RETURN(status, NULL);
    
    status = g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED);
    CHECK_STATUS_RETURN(status, NULL);
    
    // 创建会话
    status = g_ort->CreateSession(ctx->env, model_path, session_options, &ctx->session);
    CHECK_STATUS_RETURN(status, NULL);
    
    g_ort->ReleaseSessionOptions(session_options);
    
    // 创建内存信息
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ctx->memory_info);
    CHECK_STATUS_RETURN(status, NULL);
    
    // 获取输入输出信息
    status = g_ort->SessionGetInputCount(ctx->session, &ctx->num_inputs);
    CHECK_STATUS_RETURN(status, NULL);
    
    status = g_ort->SessionGetOutputCount(ctx->session, &ctx->num_outputs);
    CHECK_STATUS_RETURN(status, NULL);
    
    // 获取默认分配器
    status = g_ort->GetAllocatorWithDefaultOptions(&ctx->allocator);
    CHECK_STATUS_RETURN(status, NULL);
    
    // 获取输入输出名称
    ctx->input_names = (char**)malloc(ctx->num_inputs * sizeof(char*));
    ctx->output_names = (char**)malloc(ctx->num_outputs * sizeof(char*));
    
    for (size_t i = 0; i < ctx->num_inputs; i++) {
        status = g_ort->SessionGetInputName(ctx->session, i, ctx->allocator, &ctx->input_names[i]);
        CHECK_STATUS_RETURN(status, NULL);
    }
    
    for (size_t i = 0; i < ctx->num_outputs; i++) {
        status = g_ort->SessionGetOutputName(ctx->session, i, ctx->allocator, &ctx->output_names[i]);
        CHECK_STATUS_RETURN(status, NULL);
    }
    
    printf("✓ ONNX Runtime 初始化成功\n");
    printf("✓ 模型加载成功: %s\n", model_path);
    
    return (InferenceHandle)ctx;
}

void inference_destroy(InferenceHandle handle) {
    if (!handle) return;
    
    InferenceContext* ctx = (InferenceContext*)handle;
    
    if (ctx->session) {
        g_ort->ReleaseSession(ctx->session);
    }
    
    if (ctx->memory_info) {
        g_ort->ReleaseMemoryInfo(ctx->memory_info);
    }
    
    if (ctx->env) {
        g_ort->ReleaseEnv(ctx->env);
    }
    
    if (ctx->input_names && ctx->allocator) {
        for (size_t i = 0; i < ctx->num_inputs; i++) {
            if (ctx->input_names[i]) {
                (void)ctx->ort_api->AllocatorFree(ctx->allocator, ctx->input_names[i]);
            }
        }
        free(ctx->input_names);
    }
    
    if (ctx->output_names && ctx->allocator) {
        for (size_t i = 0; i < ctx->num_outputs; i++) {
            if (ctx->output_names[i]) {
                (void)ctx->ort_api->AllocatorFree(ctx->allocator, ctx->output_names[i]);
            }
        }
        free(ctx->output_names);
    }
    
    if (ctx->model_path) {
        free(ctx->model_path);
    }
    
    free(ctx);
}

int inference_run_single(InferenceHandle handle, int sample_id, int original_idx, 
                        int true_label, float* image_data, InferenceResult* result) {
    if (!handle || !image_data || !result) {
        return INFERENCE_ERROR_DATA;
    }
    
    InferenceContext* ctx = (InferenceContext*)handle;
    clock_t start_time = clock();
    
    // 复制输入数据（避免修改原始数据）
    float* input_data = (float*)malloc(28 * 28 * sizeof(float));
    if (!input_data) {
        return INFERENCE_ERROR_MEMORY;
    }
    memcpy(input_data, image_data, 28 * 28 * sizeof(float));
    
    // 预处理
    preprocess_image(input_data, 28 * 28);
    
    // 创建输入tensor
    int64_t input_shape[] = {1, 1, 28, 28};
    size_t input_shape_len = 4;
    
    OrtValue* input_tensor = NULL;
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(
        ctx->memory_info,
        input_data,
        28 * 28 * sizeof(float),
        input_shape,
        input_shape_len,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    );
    
    if (status != NULL) {
        g_ort->ReleaseStatus(status);
        free(input_data);
        return INFERENCE_ERROR_RUNTIME;
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
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(input_tensor);
        free(input_data);
        return INFERENCE_ERROR_RUNTIME;
    }
    
    // 获取输出数据
    float* output_data;
    status = g_ort->GetTensorMutableData(outputs[0], (void**)&output_data);
    if (status != NULL) {
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseValue(outputs[0]);
        free(input_data);
        return INFERENCE_ERROR_RUNTIME;
    }
    
    // 应用softmax并找到预测类别
    float probabilities[10];
    softmax(output_data, probabilities, 10);
    
    // 找到最大概率的类别
    result->sample_id = sample_id;
    result->original_mnist_index = original_idx;
    result->true_label = true_label;
    result->predicted_class = 0;
    result->confidence = probabilities[0];
    
    for (int i = 1; i < 10; i++) {
        if (probabilities[i] > result->confidence) {
            result->confidence = probabilities[i];
            result->predicted_class = i;
        }
    }
    
    // 检查是否正确
    result->is_correct = (result->predicted_class == result->true_label);
    
    // 计算推理时间
    clock_t end_time = clock();
    result->inference_time_ms = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    // 释放资源
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(outputs[0]);
    free(input_data);
    
    return INFERENCE_SUCCESS;
}

int inference_run_batch(InferenceHandle handle, MNISTTestData* test_data, 
                       InferenceResult* results, int num_samples) {
    if (!handle || !test_data || !results) {
        return INFERENCE_ERROR_DATA;
    }
    
    int correct_predictions = 0;
    
    for (int i = 0; i < num_samples; i++) {
        if (inference_run_single(handle, i, test_data->original_indices[i], 
                               test_data->labels[i], test_data->images[i], &results[i]) == INFERENCE_SUCCESS) {
            if (results[i].is_correct) {
                correct_predictions++;
            }
        } else {
            printf("样本 %d 推理失败\n", i);
        }
    }
    
    return correct_predictions;
}

int mnist_load_test_data(const char* test_data_dir, MNISTTestData* data) {
    if (!test_data_dir || !data) {
        return INFERENCE_ERROR_DATA;
    }
    
    printf("🔍 加载MNIST测试数据...\n");
    
    // 构造metadata文件路径
    char metadata_path[512];
    snprintf(metadata_path, sizeof(metadata_path), "%s/metadata.json", test_data_dir);
    
    // 读取元数据文件
    FILE* metadata_file = fopen(metadata_path, "r");
    if (!metadata_file) {
        printf("❌ 无法打开元数据文件: %s\n", metadata_path);
        return INFERENCE_ERROR_DATA;
    }
    
    // 解析JSON获取样本信息
    char line[1024];
    int* labels = (int*)malloc(1000 * sizeof(int));
    int* indices = (int*)malloc(1000 * sizeof(int));
    int label_count = 0;
    int num_samples = 0;
    
    while (fgets(line, sizeof(line), metadata_file) && label_count < 1000) {
        // 获取样本数量
        if (strstr(line, "\"num_samples\":")) {
            num_samples = parse_json_int(line, "\"num_samples\":");
        }
        
        // 获取标签
        if (strstr(line, "\"true_label\":")) {
            int label = parse_json_int(line, "\"true_label\":");
            if (label >= 0) {
                labels[label_count] = label;
            }
        }
        
        // 获取原始索引
        if (strstr(line, "\"original_mnist_index\":")) {
            int index = parse_json_int(line, "\"original_mnist_index\":");
            if (index >= 0) {
                indices[label_count] = index;
                label_count++;
            }
        }
    }
    fclose(metadata_file);
    
    if (num_samples <= 0 || label_count != num_samples) {
        printf("❌ 元数据解析失败: 样本数=%d, 标签数=%d\n", num_samples, label_count);
        free(labels);
        free(indices);
        return INFERENCE_ERROR_DATA;
    }
    
    printf("样本数量: %d\n", num_samples);
    printf("解析到的标签数: %d\n", label_count);
    
    // 分配内存存储数据
    data->num_samples = num_samples;
    data->images = (float**)malloc(num_samples * sizeof(float*));
    data->labels = (int*)malloc(num_samples * sizeof(int));
    data->original_indices = (int*)malloc(num_samples * sizeof(int));
    
    // 读取图像文件
    for (int i = 0; i < num_samples; i++) {
        // 构造文件名
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/image_%03d.bin", test_data_dir, i);
        
        FILE* file = fopen(filename, "rb");
        if (!file) {
            printf("❌ 无法打开文件: %s\n", filename);
            continue;
        }
        
        // 分配图像内存并读取数据
        data->images[i] = (float*)malloc(28 * 28 * sizeof(float));
        size_t read_count = fread(data->images[i], sizeof(float), 28 * 28, file);
        fclose(file);
        
        if (read_count != 28 * 28) {
            printf("❌ 文件读取失败: %s，期望读取 %d，实际读取 %zu\n", 
                   filename, 28 * 28, read_count);
            free(data->images[i]);
            continue;
        }
        
        data->labels[i] = labels[i];
        data->original_indices[i] = indices[i];
    }
    
    // 显示标签分布
    int label_dist[10] = {0};
    for (int i = 0; i < num_samples; i++) {
        if (data->labels[i] >= 0 && data->labels[i] <= 9) {
            label_dist[data->labels[i]]++;
        }
    }
    
    printf("✅ 加载了 %d 个测试样本\n", num_samples);
    printf("标签分布: [");
    for (int i = 0; i < 10; i++) {
        printf("%d", label_dist[i]);
        if (i < 9) printf(" ");
    }
    printf("]\n");
    
    free(labels);
    free(indices);
    return INFERENCE_SUCCESS;
}

void mnist_free_test_data(MNISTTestData* data) {
    if (!data) return;
    
    if (data->images) {
        for (int i = 0; i < data->num_samples; i++) {
            if (data->images[i]) {
                free(data->images[i]);
            }
        }
        free(data->images);
        data->images = NULL;
    }
    
    if (data->labels) {
        free(data->labels);
        data->labels = NULL;
    }
    
    if (data->original_indices) {
        free(data->original_indices);
        data->original_indices = NULL;
    }
    
    data->num_samples = 0;
}

void inference_save_results(InferenceResult* results, int num_samples, 
                           double total_time, int correct_predictions,
                           const char* output_path, const char* platform_name) {
    if (!results || !output_path || !platform_name) return;
    
    FILE* file = fopen(output_path, "w");
    if (file == NULL) {
        printf("警告: 无法打开结果文件进行写入: %s\n", output_path);
        return;
    }
    
    double accuracy = (double)correct_predictions / num_samples;
    double avg_time = total_time / num_samples;
    double fps = 1000.0 / avg_time;
    
    fprintf(file, "%s 统一 ONNX Runtime C API 推理结果\n", platform_name);
    fprintf(file, "==========================================\n");
    fprintf(file, "平台: %s\n", platform_name);
    fprintf(file, "总样本数: %d\n", num_samples);
    fprintf(file, "正确预测: %d\n", correct_predictions);
    fprintf(file, "准确率: %.2f%%\n", accuracy * 100);
    fprintf(file, "平均推理时间: %.2f ms\n", avg_time);
    fprintf(file, "推理速度: %.1f FPS\n", fps);
    fprintf(file, "\n样本详细结果:\n");
    
    for (int i = 0; i < num_samples; i++) {
        fprintf(file, "样本 %3d: 真实=%d, 预测=%d, 置信度=%.3f, 时间=%.2f ms, %s\n",
                results[i].sample_id,
                results[i].true_label,
                results[i].predicted_class,
                results[i].confidence,
                results[i].inference_time_ms,
                results[i].is_correct ? "正确" : "错误");
    }
    
    fclose(file);
    printf("✓ 结果已保存到 %s\n", output_path);
}

void inference_print_statistics(InferenceResult* results, int num_samples, 
                               const char* platform_name) {
    if (!results || !platform_name) return;
    
    double total_time = 0.0;
    int correct_predictions = 0;
    
    for (int i = 0; i < num_samples; i++) {
        total_time += results[i].inference_time_ms;
        if (results[i].is_correct) {
            correct_predictions++;
        }
    }
    
    double avg_time = total_time / num_samples;
    double accuracy = (double)correct_predictions / num_samples;
    int wrong_count = num_samples - correct_predictions;
    
    printf("\n=== %s 推理结果统计 ===\n", platform_name);
    printf("总样本数: %d\n", num_samples);
    printf("正确预测: %d\n", correct_predictions);
    printf("准确率: %.2f%%\n", accuracy * 100);
    printf("平均推理时间: %.2f ms\n", avg_time);
    printf("推理速度: %.1f FPS\n", 1000.0 / avg_time);
    
    // 显示错误样本
    if (wrong_count > 0) {
        printf("\n❌ 错误预测样本 (%d 个):\n", wrong_count);
        int shown = 0;
        for (int i = 0; i < num_samples && shown < 5; i++) {
            if (!results[i].is_correct) {
                printf("  样本 %3d: 真实=%d, 预测=%d, 置信度=%.3f, 时间=%.2f ms\n",
                       results[i].sample_id, results[i].true_label, 
                       results[i].predicted_class, results[i].confidence, results[i].inference_time_ms);
                shown++;
            }
        }
        if (wrong_count > 5) {
            printf("  ... 还有 %d 个错误样本\n", wrong_count - 5);
        }
    }
} 