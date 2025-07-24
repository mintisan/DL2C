#ifndef C_INFERENCE_LIB_H
#define C_INFERENCE_LIB_H

#ifdef __cplusplus
extern "C" {
#endif

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

// MNIST测试数据结构体
typedef struct {
    float** images;
    int* labels;
    int* original_indices;
    int num_samples;
} MNISTTestData;

// 推理引擎句柄（不透明指针）
typedef struct InferenceContext* InferenceHandle;

// === 核心API接口 ===

/**
 * 初始化推理引擎
 * @param model_path ONNX模型文件路径
 * @return 推理引擎句柄，失败返回NULL
 */
InferenceHandle inference_create(const char* model_path);

/**
 * 销毁推理引擎
 * @param handle 推理引擎句柄
 */
void inference_destroy(InferenceHandle handle);

/**
 * 单次推理
 * @param handle 推理引擎句柄
 * @param sample_id 样本ID
 * @param original_idx 原始MNIST索引
 * @param true_label 真实标签
 * @param image_data 图像数据 (28x28 float数组)
 * @param result 推理结果输出
 * @return 0成功，-1失败
 */
int inference_run_single(InferenceHandle handle, int sample_id, int original_idx, 
                        int true_label, float* image_data, InferenceResult* result);

/**
 * 批量推理测试
 * @param handle 推理引擎句柄
 * @param test_data 测试数据
 * @param results 结果数组输出
 * @param num_samples 样本数量
 * @return 正确预测数量，-1失败
 */
int inference_run_batch(InferenceHandle handle, MNISTTestData* test_data, 
                       InferenceResult* results, int num_samples);

// === 数据加载API ===

/**
 * 加载MNIST测试数据
 * @param test_data_dir 测试数据目录路径
 * @param data 输出的测试数据结构
 * @return 0成功，-1失败
 */
int mnist_load_test_data(const char* test_data_dir, MNISTTestData* data);

/**
 * 释放MNIST测试数据
 * @param data 测试数据结构
 */
void mnist_free_test_data(MNISTTestData* data);

// === 工具函数API ===

/**
 * 保存推理结果到文件
 * @param results 推理结果数组
 * @param num_samples 样本数量
 * @param total_time 总推理时间
 * @param correct_predictions 正确预测数量
 * @param output_path 输出文件路径
 * @param platform_name 平台名称
 */
void inference_save_results(InferenceResult* results, int num_samples, 
                           double total_time, int correct_predictions,
                           const char* output_path, const char* platform_name);

/**
 * 计算统计信息并打印
 * @param results 推理结果数组
 * @param num_samples 样本数量
 * @param platform_name 平台名称
 */
void inference_print_statistics(InferenceResult* results, int num_samples, 
                               const char* platform_name);

// === 错误码定义 ===
#define INFERENCE_SUCCESS           0
#define INFERENCE_ERROR_INIT       -1
#define INFERENCE_ERROR_MODEL      -2
#define INFERENCE_ERROR_DATA       -3
#define INFERENCE_ERROR_RUNTIME    -4
#define INFERENCE_ERROR_MEMORY     -5

#ifdef __cplusplus
}
#endif

#endif // C_INFERENCE_LIB_H 