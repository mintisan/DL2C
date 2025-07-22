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

// MNIST测试数据结构体
typedef struct {
    float** images;           // 图像数据数组
    int* labels;             // 标签数组
    int* original_indices;   // 原始MNIST索引数组
    int num_samples;         // 样本数量
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

// 初始化推理上下文
int init_inference_context(InferenceContext* ctx, const char* model_path) {
    printf("初始化ONNX Runtime C API推理引擎...\n");
    
    // 获取ORT API
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ctx->ort_api = g_ort;
    
    // 创建环境
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "CInferenceMNIST", &ctx->env);
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

// 简单的JSON解析函数，提取整数值
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

// 加载MNIST测试数据
int load_mnist_test_data(MNISTTestData* data) {
    printf("🔍 加载MNIST测试数据...\n");
    
    // 读取元数据文件
    FILE* metadata_file = fopen("../../test_data_mnist/metadata.json", "r");
    if (!metadata_file) {
        printf("❌ 无法打开元数据文件\n");
        return -1;
    }
    
    // 解析JSON获取样本信息
    char line[1024];
    int* labels = (int*)malloc(1000 * sizeof(int));  // 预分配空间
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
    
    // 读取图像文件
    for (int i = 0; i < num_samples; i++) {
        // 构造文件名
        char filename[256];
        snprintf(filename, sizeof(filename), "../../test_data_mnist/image_%03d.bin", i);
        
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
    result->original_mnist_index = original_idx;
    result->true_label = true_label;
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

// 释放推理结果
void free_inference_result(InferenceResult* result) {
    if (result->probabilities) {
        free(result->probabilities);
        result->probabilities = NULL;
    }
}

// 保存结果到JSON文件
void save_results_to_json(InferenceResult* results, size_t num_results, double avg_time, double accuracy, int wrong_count) {
    // 创建目录
    system("mkdir -p ../../results 2>/dev/null || mkdir ..\\..\\results 2>nul || true");
    
    FILE* file = fopen("../../results/c_inference_mnist_results.json", "w");
    if (!file) {
        printf("无法创建结果文件\n");
        return;
    }
    
    fprintf(file, "{\n");
    fprintf(file, "  \"platform\": \"C\",\n");
    fprintf(file, "  \"framework\": \"ONNX Runtime C API\",\n");
    fprintf(file, "  \"test_type\": \"real_mnist_data\",\n");
    fprintf(file, "  \"data_source\": \"MNIST test set subset\",\n");
    fprintf(file, "  \"summary\": {\n");
    fprintf(file, "    \"accuracy\": %.4f,\n", accuracy);
    fprintf(file, "    \"average_inference_time_ms\": %.2f,\n", avg_time);
    fprintf(file, "    \"fps\": %.1f,\n", 1000.0 / avg_time);
    fprintf(file, "    \"total_samples\": %zu,\n", num_results);
    
    int correct = 0;
    for (size_t i = 0; i < num_results; i++) {
        if (results[i].is_correct) correct++;
    }
    fprintf(file, "    \"correct_predictions\": %d,\n", correct);
    fprintf(file, "    \"wrong_predictions\": %d\n", wrong_count);
    
    fprintf(file, "  },\n");
    fprintf(file, "  \"results\": [\n");
    
    for (size_t i = 0; i < num_results; i++) {
        fprintf(file, "    {\n");
        fprintf(file, "      \"sample_id\": %d,\n", results[i].sample_id);
        fprintf(file, "      \"original_mnist_index\": %d,\n", results[i].original_mnist_index);
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
    
    printf("结果已保存到: ../../results/c_inference_mnist_results.json\n");
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
    printf("=== C ONNX推理测试 (真实MNIST数据) ===\n");
    
    InferenceContext ctx = {0};
    const char* model_path = "../../models/mnist_model.onnx";
    
    // 初始化推理上下文
    if (init_inference_context(&ctx, model_path) != 0) {
        printf("初始化失败\n");
        return -1;
    }
    
    // 加载MNIST测试数据
    MNISTTestData test_data = {0};
    if (load_mnist_test_data(&test_data) != 0) {
        printf("加载测试数据失败\n");
        cleanup_inference_context(&ctx);
        return -1;
    }
    
    printf("\n开始推理 %d 个样本...\n", test_data.num_samples);
    
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
    
    printf("\n=== 推理结果统计 ===\n");
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
                printf("  样本 %3d: 真实=%d, 预测=%d, 置信度=%.3f\n",
                       results[i].sample_id, results[i].true_label, 
                       results[i].predicted_class, results[i].confidence);
                shown++;
            }
        }
        if (wrong_count > 5) {
            printf("  ... 还有 %d 个错误样本\n", wrong_count - 5);
        }
    }
    
    // 保存结果
    save_results_to_json(results, test_data.num_samples, avg_time, accuracy, wrong_count);
    
    // 清理资源
    for (int i = 0; i < test_data.num_samples; i++) {
        free_inference_result(&results[i]);
    }
    free(results);
    free_mnist_test_data(&test_data);
    cleanup_inference_context(&ctx);
    
    printf("\n✅ C推理测试完成\n");
    
    return 0;
} 