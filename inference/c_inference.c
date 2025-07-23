#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "onnxruntime_c_api.h"

// 平台特定的路径配置
#ifdef __ANDROID__
    #define MODEL_PATH "/data/local/tmp/mnist_onnx/models/mnist_model.onnx"
    #define RESULTS_PATH "/data/local/tmp/mnist_onnx/results/android_c_results.txt"
    #define TEST_DATA_DIR "/data/local/tmp/mnist_onnx/test_data"
    #define PLATFORM_NAME "Android"
#else
    #define MODEL_PATH "../models/mnist_model.onnx"
    #define RESULTS_PATH "../results/macos_c_results.txt"
    #define TEST_DATA_DIR "../test_data"
    #define PLATFORM_NAME "macOS"
#endif

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
    double inference_time_ms;
    int is_correct;
} InferenceResult;

// MNIST测试数据结构体（统一版本）
typedef struct {
    float** images;
    int* labels;
    int* original_indices;
    int num_samples;
} MNISTTestData;

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

// 简单的JSON解析函数
int parse_json_int(const char* line, const char* key) {
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

// 初始化推理上下文
int init_inference_context(InferenceContext* ctx, const char* model_path) {
    printf("初始化ONNX Runtime C API推理引擎...\n");
    
    // 获取ORT API
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ctx->ort_api = g_ort;
    
    // 创建环境
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "CInferenceUnified", &ctx->env);
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
    
    printf("✓ ONNX Runtime 初始化成功\n");
    printf("✓ 模型加载成功: %s\n", model_path);
    
    return 0;
}

// 预处理函数（与原始版本保持一致）
void preprocess_image(float* input_data, size_t data_size) {
    // MNIST标准化参数
    const float mean = 0.1307f;
    const float std = 0.3081f;
    
    // 标准化: (pixel - mean) / std
    for (size_t i = 0; i < data_size; i++) {
        input_data[i] = (input_data[i] - mean) / std;
    }
}

// Softmax函数（与原始版本保持一致）
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

// 加载MNIST测试数据（使用真实数据文件，与原始版本逻辑一致）
int load_mnist_test_data(MNISTTestData* data) {
    printf("🔍 加载MNIST测试数据...\n");
    
    // 构造metadata文件路径
    char metadata_path[512];
    snprintf(metadata_path, sizeof(metadata_path), "%s/metadata.json", TEST_DATA_DIR);
    
    // 读取元数据文件
    FILE* metadata_file = fopen(metadata_path, "r");
    if (!metadata_file) {
        printf("❌ 无法打开元数据文件: %s\n", metadata_path);
        return -1;
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
        return -1;
    }
    
    printf("样本数量: %d\n", num_samples);
    printf("解析到的标签数: %d\n", label_count);
    
    // 分配内存存储数据
    data->num_samples = num_samples;
    data->images = (float**)malloc(num_samples * sizeof(float*));
    data->labels = (int*)malloc(num_samples * sizeof(int));
    data->original_indices = (int*)malloc(num_samples * sizeof(int));
    
    // 读取图像文件（与原始c_inference_mnist.c一致）
    for (int i = 0; i < num_samples; i++) {
        // 构造文件名
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/image_%03d.bin", TEST_DATA_DIR, i);
        
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
    return 0;
}

// 执行推理
int run_inference(InferenceContext* ctx, int sample_id, int original_idx, int true_label, 
                  float* image_data, InferenceResult* result) {
    clock_t start_time = clock();
    
    // 复制输入数据（避免修改原始数据）
    float* input_data = (float*)malloc(28 * 28 * sizeof(float));
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
    
    return 0;
}

// 释放MNIST测试数据
void free_mnist_test_data(MNISTTestData* data) {
    if (data->images) {
        for (int i = 0; i < data->num_samples; i++) {
            if (data->images[i]) {
                free(data->images[i]);
            }
        }
        free(data->images);
    }
    
    if (data->labels) {
        free(data->labels);
    }
    
    if (data->original_indices) {
        free(data->original_indices);
    }
}

// 保存结果到文件
void save_results(InferenceResult* results, int num_samples, double total_time, int correct_predictions) {
    FILE* file = fopen(RESULTS_PATH, "w");
    if (file == NULL) {
        printf("警告: 无法打开结果文件进行写入\n");
        return;
    }
    
    double accuracy = (double)correct_predictions / num_samples;
    double avg_time = total_time / num_samples;
    double fps = 1000.0 / avg_time;
    
    fprintf(file, "%s 统一 ONNX Runtime C API 推理结果\n", PLATFORM_NAME);
    fprintf(file, "==========================================\n");
    fprintf(file, "平台: %s\n", PLATFORM_NAME);
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
    printf("✓ 结果已保存到 %s\n", RESULTS_PATH);
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
                (void)ctx->ort_api->AllocatorFree(ctx->allocator, ctx->input_names[i]);
            }
        }
        free(ctx->input_names);
        ctx->input_names = NULL;
    }
    
    if (ctx->output_names && ctx->allocator) {
        for (size_t i = 0; i < ctx->num_outputs; i++) {
            if (ctx->output_names[i]) {
                (void)ctx->ort_api->AllocatorFree(ctx->allocator, ctx->output_names[i]);
            }
        }
        free(ctx->output_names);
        ctx->output_names = NULL;
    }
}

// 主函数
int main() {
    printf("启动 %s 统一 ONNX Runtime C API MNIST 推理程序...\n", PLATFORM_NAME);
    
    InferenceContext ctx = {0};
    
    // 初始化推理上下文
    if (init_inference_context(&ctx, MODEL_PATH) != 0) {
        printf("初始化失败\n");
        return -1;
    }
    
    printf("\n=== 开始 %s 统一推理测试 ===\n", PLATFORM_NAME);
    
    // 加载MNIST测试数据（使用真实数据）
    MNISTTestData test_data = {0};
    if (load_mnist_test_data(&test_data) != 0) {
        printf("加载测试数据失败\n");
        cleanup_inference_context(&ctx);
        return -1;
    }
    
    printf("开始推理 %d 个样本...\n", test_data.num_samples);
    
    // 分配内存存储结果
    InferenceResult* results = (InferenceResult*)malloc(test_data.num_samples * sizeof(InferenceResult));
    
    double total_time = 0.0;
    int correct_predictions = 0;
    
    // 执行推理
    for (int i = 0; i < test_data.num_samples; i++) {
        if (run_inference(&ctx, i, test_data.original_indices[i], test_data.labels[i], 
                         test_data.images[i], &results[i]) == 0) {
            total_time += results[i].inference_time_ms;
            if (results[i].is_correct) {
                correct_predictions++;
            }
            
            // 显示进度（每10个样本）
            if ((i + 1) % 10 == 0) {
                double current_accuracy = (double)correct_predictions / (i + 1) * 100;
                printf("完成 %3d/%d 样本，当前准确率: %.1f%%\n", 
                       i + 1, test_data.num_samples, current_accuracy);
            }
        } else {
            printf("样本 %d 推理失败\n", i);
        }
    }
    
    // 计算统计信息
    double avg_time = total_time / test_data.num_samples;
    double accuracy = (double)correct_predictions / test_data.num_samples;
    int wrong_count = test_data.num_samples - correct_predictions;
    
    printf("\n=== %s 推理结果统计 ===\n", PLATFORM_NAME);
    printf("总样本数: %d\n", test_data.num_samples);
    printf("正确预测: %d\n", correct_predictions);
    printf("准确率: %.2f%%\n", accuracy * 100);
    printf("平均推理时间: %.2f ms\n", avg_time);
    printf("推理速度: %.1f FPS\n", 1000.0 / avg_time);
    
    // 显示错误样本
    if (wrong_count > 0) {
        printf("\n❌ 错误预测样本 (%d 个):\n", wrong_count);
        int shown = 0;
        for (int i = 0; i < test_data.num_samples && shown < 5; i++) {
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
    
    // 保存结果
    save_results(results, test_data.num_samples, total_time, correct_predictions);
    
    // 清理资源
    free(results);
    free_mnist_test_data(&test_data);
    cleanup_inference_context(&ctx);
    
    printf("\n✅ %s 统一推理测试完成\n", PLATFORM_NAME);
    
    return 0;
} 