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
    int true_label;
    int predicted_class;
    float confidence;
    float* probabilities;
    size_t prob_count;
    double inference_time_ms;
    int is_correct;
} InferenceResult;

// 测试数据结构体
typedef struct {
    float* image_data;    // 28*28 像素数据
    int true_label;       // 真实标签
} TestSample;

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

// 初始化推理上下文
int init_inference_context(InferenceContext* ctx, const char* model_path) {
    printf("初始化ONNX Runtime C API推理引擎...\n");
    
    // 获取ORT API
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ctx->ort_api = g_ort;
    
    // 创建环境
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "CInferenceCommon", &ctx->env);
    CHECK_STATUS(status);
    
    // 创建会话选项
    OrtSessionOptions* session_options;
    status = g_ort->CreateSessionOptions(&session_options);
    CHECK_STATUS(status);
    
    status = g_ort->SetIntraOpNumThreads(session_options, 1);
    CHECK_STATUS(status);
    
    status = g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED);
    CHECK_STATUS(status);
    
    // 创建会话
    status = g_ort->CreateSession(ctx->env, model_path, session_options, &ctx->session);
    CHECK_STATUS(status);
    
    g_ort->ReleaseSessionOptions(session_options);
    
    // 创建内存信息
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ctx->memory_info);
    CHECK_STATUS(status);
    
    // 获取输入输出信息
    status = g_ort->SessionGetInputCount(ctx->session, &ctx->num_inputs);
    CHECK_STATUS(status);
    
    status = g_ort->SessionGetOutputCount(ctx->session, &ctx->num_outputs);
    CHECK_STATUS(status);
    
    // 获取默认分配器
    status = g_ort->GetAllocatorWithDefaultOptions(&ctx->allocator);
    CHECK_STATUS(status);
    
    // 获取输入输出名称
    ctx->input_names = (char**)malloc(ctx->num_inputs * sizeof(char*));
    ctx->output_names = (char**)malloc(ctx->num_outputs * sizeof(char*));
    
    for (size_t i = 0; i < ctx->num_inputs; i++) {
        status = g_ort->SessionGetInputName(ctx->session, i, ctx->allocator, &ctx->input_names[i]);
        CHECK_STATUS(status);
    }
    
    for (size_t i = 0; i < ctx->num_outputs; i++) {
        status = g_ort->SessionGetOutputName(ctx->session, i, ctx->allocator, &ctx->output_names[i]);
        CHECK_STATUS(status);
    }
    
    printf("✅ ONNX Runtime C API初始化成功\n");
    return 0;
}

// 加载共同测试数据
int load_common_test_data(TestSample** samples, int* num_samples) {
    printf("🔍 加载共同测试数据...\n");
    
    // 读取元数据获取标签
    FILE* metadata_file = fopen("../../test_data/metadata.json", "r");
    if (!metadata_file) {
        printf("❌ 无法打开元数据文件\n");
        return -1;
    }
    
    int labels[10];
    int label_count = 0;
    char line[1024];
    
    // 简单解析JSON获取标签
    while (fgets(line, sizeof(line), metadata_file) && label_count < 10) {
        char* pos = strstr(line, "\"true_label\":");
        if (pos) {
            pos += 13; // 跳过 "true_label":
            while (*pos == ' ' || *pos == '"') pos++;
            labels[label_count] = atoi(pos);
            label_count++;
        }
    }
    fclose(metadata_file);
    
    if (label_count != 10) {
        printf("❌ 标签解析失败，只解析到 %d 个标签\n", label_count);
        return -1;
    }
    
    // 分配内存存储测试样本
    *num_samples = 10;
    *samples = (TestSample*)malloc(10 * sizeof(TestSample));
    
    // 读取二进制图像文件
    for (int i = 0; i < 10; i++) {
        char filename[256];
        snprintf(filename, sizeof(filename), "../../test_data/sample_%02d.bin", i);
        
        FILE* file = fopen(filename, "rb");
        if (!file) {
            printf("❌ 无法打开文件: %s\n", filename);
            return -1;
        }
        
        // 分配内存并读取28*28个float32值
        (*samples)[i].image_data = (float*)malloc(28 * 28 * sizeof(float));
        (*samples)[i].true_label = labels[i];
        
        size_t read_count = fread((*samples)[i].image_data, sizeof(float), 28 * 28, file);
        fclose(file);
        
        if (read_count != 28 * 28) {
            printf("❌ 文件读取失败: %s，期望读取 %d，实际读取 %zu\n", 
                   filename, 28 * 28, read_count);
            return -1;
        }
        
        printf("样本 %d: 真实标签=%d\n", i, labels[i]);
    }
    
    printf("✅ 加载了 %d 个测试样本\n", *num_samples);
    return 0;
}

// 预处理函数
void preprocess_image(float* input_data, size_t data_size) {
    // MNIST标准化参数
    const float mean = 0.1307f;
    const float std = 0.3081f;
    
    // 标准化: (pixel - mean) / std
    for (size_t i = 0; i < data_size; i++) {
        input_data[i] = (input_data[i] - mean) / std;
    }
}

// Softmax函数
void softmax(float* input, float* output, size_t size) {
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

// 执行推理
int run_inference(InferenceContext* ctx, int sample_id, TestSample* sample, InferenceResult* result) {
    clock_t start_time = clock();
    
    // 复制输入数据（避免修改原始数据）
    float* input_data = (float*)malloc(28 * 28 * sizeof(float));
    memcpy(input_data, sample->image_data, 28 * 28 * sizeof(float));
    
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
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(input_tensor);
        free(input_data);
        return -1;
    }
    
    // 获取输出数据
    float* output_data;
    status = g_ort->GetTensorMutableData(outputs[0], (void**)&output_data);
    if (status != NULL) {
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseValue(outputs[0]);
        free(input_data);
        return -1;
    }
    
    // 获取输出维度信息
    OrtTensorTypeAndShapeInfo* output_info;
    status = g_ort->GetTensorTypeAndShape(outputs[0], &output_info);
    if (status != NULL) {
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseValue(outputs[0]);
        free(input_data);
        return -1;
    }
    
    size_t output_count;
    status = g_ort->GetTensorShapeElementCount(output_info, &output_count);
    if (status != NULL) {
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseTensorTypeAndShapeInfo(output_info);
        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseValue(outputs[0]);
        free(input_data);
        return -1;
    }
    
    // 应用softmax并找到预测类别
    result->sample_id = sample_id;
    result->true_label = sample->true_label;
    result->prob_count = output_count;
    result->probabilities = (float*)malloc(output_count * sizeof(float));
    
    // 计算softmax
    softmax(output_data, result->probabilities, output_count);
    
    // 找到最大概率的类别
    result->predicted_class = 0;
    result->confidence = result->probabilities[0];
    for (size_t i = 1; i < output_count; i++) {
        if (result->probabilities[i] > result->confidence) {
            result->confidence = result->probabilities[i];
            result->predicted_class = (int)i;
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
    g_ort->ReleaseTensorTypeAndShapeInfo(output_info);
    free(input_data);
    
    return 0;
}

// 释放测试样本
void free_test_samples(TestSample* samples, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        free(samples[i].image_data);
    }
    free(samples);
}

// 释放推理结果
void free_inference_result(InferenceResult* result) {
    if (result->probabilities) {
        free(result->probabilities);
        result->probabilities = NULL;
    }
}

// 保存结果到JSON文件
void save_results_to_json(InferenceResult* results, size_t num_results, double avg_time, double accuracy) {
    // 创建目录
    system("mkdir -p ../../results 2>/dev/null || mkdir ..\\..\\results 2>nul || true");
    
    FILE* file = fopen("../../results/c_inference_common_results.json", "w");
    if (!file) {
        printf("无法创建结果文件\n");
        return;
    }
    
    fprintf(file, "{\n");
    fprintf(file, "  \"platform\": \"C\",\n");
    fprintf(file, "  \"framework\": \"ONNX Runtime C API\",\n");
    fprintf(file, "  \"test_type\": \"common_data\",\n");
    fprintf(file, "  \"summary\": {\n");
    fprintf(file, "    \"accuracy\": %.4f,\n", accuracy);
    fprintf(file, "    \"average_inference_time_ms\": %.2f,\n", avg_time);
    fprintf(file, "    \"fps\": %.1f,\n", 1000.0 / avg_time);
    fprintf(file, "    \"total_samples\": %zu,\n", num_results);
    
    int correct = 0;
    for (size_t i = 0; i < num_results; i++) {
        if (results[i].is_correct) correct++;
    }
    fprintf(file, "    \"correct_predictions\": %d\n", correct);
    
    fprintf(file, "  },\n");
    fprintf(file, "  \"results\": [\n");
    
    for (size_t i = 0; i < num_results; i++) {
        fprintf(file, "    {\n");
        fprintf(file, "      \"sample_id\": %d,\n", results[i].sample_id);
        fprintf(file, "      \"true_label\": %d,\n", results[i].true_label);
        fprintf(file, "      \"predicted_class\": %d,\n", results[i].predicted_class);
        fprintf(file, "      \"confidence\": %.4f,\n", results[i].confidence);
        fprintf(file, "      \"inference_time_ms\": %.2f,\n", results[i].inference_time_ms);
        fprintf(file, "      \"is_correct\": %s\n", results[i].is_correct ? "true" : "false");
        fprintf(file, "    }");
        if (i < num_results - 1) fprintf(file, ",");
        fprintf(file, "\n");
    }
    
    fprintf(file, "  ]\n");
    fprintf(file, "}\n");
    fclose(file);
    
    printf("结果已保存到: ../../results/c_inference_common_results.json\n");
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
    
    if (ctx->input_names && ctx->allocator) {
        for (size_t i = 0; i < ctx->num_inputs; i++) {
            if (ctx->input_names[i]) {
                ctx->ort_api->AllocatorFree(ctx->allocator, ctx->input_names[i]);
            }
        }
        free(ctx->input_names);
        ctx->input_names = NULL;
    }
    
    if (ctx->output_names && ctx->allocator) {
        for (size_t i = 0; i < ctx->num_outputs; i++) {
            if (ctx->output_names[i]) {
                ctx->ort_api->AllocatorFree(ctx->allocator, ctx->output_names[i]);
            }
        }
        free(ctx->output_names);
        ctx->output_names = NULL;
    }
}

// 主函数
int main() {
    printf("=== C ONNX推理测试 (共同数据) ===\n");
    
    InferenceContext ctx = {0};
    const char* model_path = "../../models/mnist_model.onnx";
    
    // 初始化推理上下文
    if (init_inference_context(&ctx, model_path) != 0) {
        printf("初始化失败\n");
        return -1;
    }
    
    // 加载测试数据
    TestSample* test_samples = NULL;
    int num_samples = 0;
    
    if (load_common_test_data(&test_samples, &num_samples) != 0) {
        printf("加载测试数据失败\n");
        cleanup_inference_context(&ctx);
        return -1;
    }
    
    printf("\n开始推理 %d 个样本...\n", num_samples);
    
    // 分配内存存储结果
    InferenceResult* results = (InferenceResult*)malloc(num_samples * sizeof(InferenceResult));
    
    double total_time = 0.0;
    int correct_predictions = 0;
    
    // 执行推理
    for (int i = 0; i < num_samples; i++) {
        if (run_inference(&ctx, i, &test_samples[i], &results[i]) == 0) {
            total_time += results[i].inference_time_ms;
            if (results[i].is_correct) {
                correct_predictions++;
            }
            
            printf("样本 %2d: 真实=%d, 预测=%d, 置信度=%.4f, 时间=%.2fms, %s\n",
                   i, results[i].true_label, results[i].predicted_class, 
                   results[i].confidence, results[i].inference_time_ms,
                   results[i].is_correct ? "✓" : "✗");
        } else {
            printf("样本 %d 推理失败\n", i);
        }
    }
    
    // 计算统计信息
    double avg_time = total_time / num_samples;
    double accuracy = (double)correct_predictions / num_samples;
    
    printf("\n=== 推理结果统计 ===\n");
    printf("总样本数: %d\n", num_samples);
    printf("正确预测: %d\n", correct_predictions);
    printf("准确率: %.2f%%\n", accuracy * 100);
    printf("平均推理时间: %.2f ms\n", avg_time);
    printf("推理速度: %.1f FPS\n", 1000.0 / avg_time);
    
    // 保存结果
    save_results_to_json(results, num_samples, avg_time, accuracy);
    
    // 清理资源
    for (int i = 0; i < num_samples; i++) {
        free_inference_result(&results[i]);
    }
    free(results);
    free_test_samples(test_samples, num_samples);
    cleanup_inference_context(&ctx);
    
    printf("\n✅ C推理测试完成\n");
    
    return 0;
} 