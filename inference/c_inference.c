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
    int64_t* input_shape;
    size_t input_shape_len;
} InferenceContext;

// 推理结果结构体
typedef struct {
    int predicted_class;
    float confidence;
    float* probabilities;
    size_t prob_count;
    double inference_time_ms;
} InferenceResult;

// 错误处理宏
#define CHECK_STATUS(status) \
    if (status != NULL) { \
        const char* msg = g_ort->GetErrorMessage(status); \
        printf("错误: %s\n", msg); \
        g_ort->ReleaseStatus(status); \
        return -1; \
    }

// 全局ORT API指针
const OrtApi* g_ort = NULL;

// 初始化推理上下文
int init_inference_context(InferenceContext* ctx, const char* model_path) {
    printf("初始化ONNX Runtime C API推理引擎...\n");
    
    // 获取ORT API
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ctx->ort_api = g_ort;
    
    // 创建环境
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "MNISTInference", &ctx->env);
    CHECK_STATUS(status);
    
    // 创建会话选项
    OrtSessionOptions* session_options;
    status = g_ort->CreateSessionOptions(&session_options);
    CHECK_STATUS(status);
    
    // 设置会话选项
    status = g_ort->SetIntraOpNumThreads(session_options, 1);
    CHECK_STATUS(status);
    
    status = g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED);
    CHECK_STATUS(status);
    
    // 创建会话
    status = g_ort->CreateSession(ctx->env, model_path, session_options, &ctx->session);
    CHECK_STATUS(status);
    
    // 释放会话选项
    g_ort->ReleaseSessionOptions(session_options);
    
    // 创建内存信息
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ctx->memory_info);
    CHECK_STATUS(status);
    
    // 获取输入信息
    status = g_ort->SessionGetInputCount(ctx->session, &ctx->num_inputs);
    CHECK_STATUS(status);
    
    printf("输入节点数量: %zu\n", ctx->num_inputs);
    
    ctx->input_names = (char**)malloc(ctx->num_inputs * sizeof(char*));
    
    // 获取默认分配器并保存到context
    status = g_ort->GetAllocatorWithDefaultOptions(&ctx->allocator);
    CHECK_STATUS(status);
    
    for (size_t i = 0; i < ctx->num_inputs; i++) {
        // 直接使用ONNX Runtime返回的字符串指针
        status = g_ort->SessionGetInputName(ctx->session, i, ctx->allocator, &ctx->input_names[i]);
        CHECK_STATUS(status);
        
        printf("输入名称[%zu]: %s\n", i, ctx->input_names[i]);
        
        // 获取输入类型信息
        OrtTypeInfo* type_info;
        status = g_ort->SessionGetInputTypeInfo(ctx->session, i, &type_info);
        CHECK_STATUS(status);
        
        const OrtTensorTypeAndShapeInfo* tensor_info;
        status = g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
        CHECK_STATUS(status);
        
        // 获取输入形状
        status = g_ort->GetDimensionsCount(tensor_info, &ctx->input_shape_len);
        CHECK_STATUS(status);
        
        ctx->input_shape = (int64_t*)malloc(ctx->input_shape_len * sizeof(int64_t));
        status = g_ort->GetDimensions(tensor_info, ctx->input_shape, ctx->input_shape_len);
        CHECK_STATUS(status);
        
        printf("输入形状: [");
        for (size_t j = 0; j < ctx->input_shape_len; j++) {
            printf("%lld", (long long)ctx->input_shape[j]);
            if (j < ctx->input_shape_len - 1) printf(", ");
        }
        printf("]\n");
        
        g_ort->ReleaseTypeInfo(type_info);
    }
    
    // 获取输出信息
    status = g_ort->SessionGetOutputCount(ctx->session, &ctx->num_outputs);
    CHECK_STATUS(status);
    
    printf("输出节点数量: %zu\n", ctx->num_outputs);
    
    ctx->output_names = (char**)malloc(ctx->num_outputs * sizeof(char*));
    
    for (size_t i = 0; i < ctx->num_outputs; i++) {
        // 直接使用ONNX Runtime返回的字符串指针
        status = g_ort->SessionGetOutputName(ctx->session, i, ctx->allocator, &ctx->output_names[i]);
        CHECK_STATUS(status);
        
        printf("输出名称[%zu]: %s\n", i, ctx->output_names[i]);
    }
    
    printf("✓ ONNX Runtime C API初始化成功\n");
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
int run_inference(InferenceContext* ctx, float* input_data, size_t input_size, InferenceResult* result) {
    clock_t start_time = clock();
    
    // 预处理
    preprocess_image(input_data, input_size);
    
    // 创建输入tensor - 使用固定形状 [1, 1, 28, 28]
    int64_t input_shape_fixed[] = {1, 1, 28, 28};
    size_t input_shape_len_fixed = 4;
    
    OrtValue* input_tensor = NULL;
    OrtStatus* status = g_ort->CreateTensorWithDataAsOrtValue(
        ctx->memory_info,
        input_data,
        input_size * sizeof(float),
        input_shape_fixed,
        input_shape_len_fixed,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor
    );
    CHECK_STATUS(status);
    
    // 准备输入输出数组
    const OrtValue* inputs[] = { input_tensor };
    OrtValue* outputs[1] = { NULL };
    
    // 运行推理
    status = g_ort->Run(
        ctx->session,
        NULL,                                    // RunOptions
        (const char* const*)ctx->input_names,   // 输入名称
        inputs,                                  // 输入values
        ctx->num_inputs,                         // 输入数量
        (const char* const*)ctx->output_names,  // 输出名称
        ctx->num_outputs,                        // 输出数量
        outputs                                  // 输出values
    );
    CHECK_STATUS(status);
    
    // 获取输出数据
    float* output_data;
    status = g_ort->GetTensorMutableData(outputs[0], (void**)&output_data);
    CHECK_STATUS(status);
    
    // 获取输出维度信息
    OrtTensorTypeAndShapeInfo* output_info;
    status = g_ort->GetTensorTypeAndShape(outputs[0], &output_info);
    CHECK_STATUS(status);
    
    size_t output_count;
    status = g_ort->GetTensorShapeElementCount(output_info, &output_count);
    CHECK_STATUS(status);
    
    // 应用softmax并找到预测类别
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
    
    // 计算推理时间
    clock_t end_time = clock();
    result->inference_time_ms = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    // 释放资源
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(outputs[0]);
    g_ort->ReleaseTensorTypeAndShapeInfo(output_info);
    
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
    
    // 释放输入输出名称（使用allocator释放）
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
    
    if (ctx->input_shape) {
        free(ctx->input_shape);
        ctx->input_shape = NULL;
    }
}

// 释放推理结果
void free_inference_result(InferenceResult* result) {
    if (result->probabilities) {
        free(result->probabilities);
        result->probabilities = NULL;
    }
}

// 生成随机测试数据
void generate_random_data(float* data, size_t size) {
    srand((unsigned int)time(NULL));
    for (size_t i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX;  // 0-1范围的随机数
    }
}

// 保存结果到JSON文件
void save_results_to_json(InferenceResult* results, size_t num_results, double avg_time) {
    // 尝试创建results目录
    system("mkdir -p ../../results 2>/dev/null || mkdir ..\\..\\results 2>nul || true");
    
    FILE* file = fopen("../../results/c_inference_results.json", "w");
    if (!file) {
        printf("无法创建结果文件\n");
        return;
    }
    
    fprintf(file, "{\n");
    fprintf(file, "  \"platform\": \"C\",\n");
    fprintf(file, "  \"framework\": \"ONNX Runtime C API\",\n");
    fprintf(file, "  \"summary\": {\n");
    fprintf(file, "    \"total_samples\": %zu,\n", num_results);
    fprintf(file, "    \"average_inference_time_ms\": %.2f,\n", avg_time);
    fprintf(file, "    \"fps\": %.1f\n", 1000.0 / avg_time);
    fprintf(file, "  },\n");
    fprintf(file, "  \"results\": [\n");
    
    for (size_t i = 0; i < num_results; i++) {
        fprintf(file, "    {\n");
        fprintf(file, "      \"sample_id\": %zu,\n", i);
        fprintf(file, "      \"predicted_class\": %d,\n", results[i].predicted_class);
        fprintf(file, "      \"confidence\": %.4f,\n", results[i].confidence);
        fprintf(file, "      \"inference_time_ms\": %.2f\n", results[i].inference_time_ms);
        fprintf(file, "    }");
        if (i < num_results - 1) fprintf(file, ",");
        fprintf(file, "\n");
    }
    
    fprintf(file, "  ]\n");
    fprintf(file, "}\n");
    fclose(file);
    
    printf("结果已保存到: ../../results/c_inference_results.json\n");
}

// 主函数
int main() {
    printf("=== ONNX Runtime C API 推理测试 ===\n");
    
    InferenceContext ctx = {0};
    const char* model_path = "../../models/mnist_model.onnx";
    
    // 初始化推理上下文
    if (init_inference_context(&ctx, model_path) != 0) {
        printf("初始化失败\n");
        return -1;
    }
    
    // 测试参数
    const size_t test_samples = 10;
    const size_t input_size = 28 * 28;  // MNIST图像大小
    
    printf("\n=== 单样本推理测试 ===\n");
    
    // 分配内存存储结果
    InferenceResult* results = (InferenceResult*)malloc(test_samples * sizeof(InferenceResult));
    float* input_data = (float*)malloc(input_size * sizeof(float));
    
    double total_time = 0.0;
    
    for (size_t i = 0; i < test_samples; i++) {
        // 生成随机测试数据
        generate_random_data(input_data, input_size);
        
        // 执行推理
        if (run_inference(&ctx, input_data, input_size, &results[i]) == 0) {
            total_time += results[i].inference_time_ms;
            
            printf("样本 %2zu: 预测=%d, 置信度=%.4f, 时间=%.2fms\n",
                   i, results[i].predicted_class, results[i].confidence, results[i].inference_time_ms);
        } else {
            printf("样本 %zu 推理失败\n", i);
        }
    }
    
    // 计算统计信息
    double avg_time = total_time / test_samples;
    double min_time = results[0].inference_time_ms;
    double max_time = results[0].inference_time_ms;
    
    for (size_t i = 1; i < test_samples; i++) {
        if (results[i].inference_time_ms < min_time) {
            min_time = results[i].inference_time_ms;
        }
        if (results[i].inference_time_ms > max_time) {
            max_time = results[i].inference_time_ms;
        }
    }
    
    printf("\n=== 性能统计 ===\n");
    printf("总样本数: %zu\n", test_samples);
    printf("平均推理时间: %.2f ms\n", avg_time);
    printf("推理速度: %.1f FPS\n", 1000.0 / avg_time);
    printf("最快推理: %.2f ms\n", min_time);
    printf("最慢推理: %.2f ms\n", max_time);
    
    // 批量推理测试
    printf("\n=== 批量推理测试 ===\n");
    const size_t batch_size = 100;
    
    clock_t batch_start = clock();
    for (size_t i = 0; i < batch_size; i++) {
        generate_random_data(input_data, input_size);
        InferenceResult temp_result = {0};
        run_inference(&ctx, input_data, input_size, &temp_result);
        free_inference_result(&temp_result);
    }
    clock_t batch_end = clock();
    
    double batch_total_time = ((double)(batch_end - batch_start)) / CLOCKS_PER_SEC * 1000.0;
    double batch_avg_time = batch_total_time / batch_size;
    
    printf("批量大小: %zu\n", batch_size);
    printf("批量总时间: %.2f ms\n", batch_total_time);
    printf("平均单样本时间: %.2f ms\n", batch_avg_time);
    printf("批量推理速度: %.1f FPS\n", batch_size * 1000.0 / batch_total_time);
    
    // 保存结果
    save_results_to_json(results, test_samples, avg_time);
    
    // 清理资源
    for (size_t i = 0; i < test_samples; i++) {
        free_inference_result(&results[i]);
    }
    free(results);
    free(input_data);
    cleanup_inference_context(&ctx);
    
    printf("\n=== C API推理测试完成 ===\n");
    printf("注意: 由于使用模拟数据，无法验证准确性\n");
    printf("在实际应用中，请使用真实的MNIST测试数据\n");
    
    return 0;
} 