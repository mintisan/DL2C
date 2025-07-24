#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "c_inference_lib.h"

// 平台特定的路径配置
#ifdef __ANDROID__
    #define MODEL_PATH "/data/local/tmp/mnist_onnx/models/mnist_model.onnx"
    #define RESULTS_PATH "/data/local/tmp/mnist_onnx/results/android_c_lib_results.txt"
    #define TEST_DATA_DIR "/data/local/tmp/mnist_onnx/test_data"
    #define PLATFORM_NAME "Android"
#else
    #define MODEL_PATH "../models/mnist_model.onnx"
    #define RESULTS_PATH "../results/macos_c_lib_results.txt"
    #define TEST_DATA_DIR "../test_data"
    #define PLATFORM_NAME "macOS"
#endif

int main() {
    printf("启动 %s 统一 ONNX Runtime C库 MNIST 推理程序...\n", PLATFORM_NAME);
    
    // 显示库版本信息
    inference_print_version_info();
    printf("\n");
    
    // 创建推理引擎
    InferenceHandle inference_handle = inference_create(MODEL_PATH);
    if (!inference_handle) {
        printf("❌ 推理引擎初始化失败\n");
        return -1;
    }
    
    printf("\n=== 开始 %s 统一推理测试 ===\n", PLATFORM_NAME);
    
    // 加载MNIST测试数据
    MNISTTestData test_data = {0};
    int load_result = mnist_load_test_data(TEST_DATA_DIR, &test_data);
    if (load_result != INFERENCE_SUCCESS) {
        printf("❌ 加载测试数据失败，错误码: %d\n", load_result);
        inference_destroy(inference_handle);
        return -1;
    }
    
    printf("开始推理 %d 个样本...\n", test_data.num_samples);
    
    // 分配内存存储结果
    InferenceResult* results = (InferenceResult*)malloc(test_data.num_samples * sizeof(InferenceResult));
    if (!results) {
        printf("❌ 结果数组内存分配失败\n");
        mnist_free_test_data(&test_data);
        inference_destroy(inference_handle);
        return -1;
    }
    
    double total_time = 0.0;
    int correct_predictions = 0;
    
    // 执行批量推理
    printf("使用批量推理接口...\n");
    correct_predictions = inference_run_batch(inference_handle, &test_data, results, test_data.num_samples);
    
    if (correct_predictions < 0) {
        printf("❌ 批量推理执行失败\n");
    } else {
        // 计算总时间
        for (int i = 0; i < test_data.num_samples; i++) {
            total_time += results[i].inference_time_ms;
            
            // 显示进度（每10个样本）
            if ((i + 1) % 10 == 0) {
                // 重新计算当前的正确预测数
                int current_correct = 0;
                for (int j = 0; j <= i; j++) {
                    if (results[j].is_correct) current_correct++;
                }
                double current_accuracy = (double)current_correct / (i + 1) * 100;
                printf("完成 %3d/%d 样本，当前准确率: %.1f%%\n", 
                       i + 1, test_data.num_samples, current_accuracy);
            }
        }
        
        // 显示统计信息
        inference_print_statistics(results, test_data.num_samples, PLATFORM_NAME);
        
        // 保存结果
        inference_save_results(results, test_data.num_samples, total_time, 
                              correct_predictions, RESULTS_PATH, PLATFORM_NAME);
    }
    
    // 演示单次推理API（可选）
    printf("\n=== 演示单次推理API ===\n");
    if (test_data.num_samples > 0) {
        InferenceResult single_result;
        int single_inference_result = inference_run_single(
            inference_handle, 
            0,  // sample_id
            test_data.original_indices[0],  // original_idx
            test_data.labels[0],  // true_label
            test_data.images[0],  // image_data
            &single_result
        );
        
        if (single_inference_result == INFERENCE_SUCCESS) {
            printf("单次推理演示成功:\n");
            printf("  样本ID: %d\n", single_result.sample_id);
            printf("  真实标签: %d\n", single_result.true_label);
            printf("  预测结果: %d\n", single_result.predicted_class);
            printf("  置信度: %.3f\n", single_result.confidence);
            printf("  推理时间: %.2f ms\n", single_result.inference_time_ms);
            printf("  预测是否正确: %s\n", single_result.is_correct ? "正确" : "错误");
        } else {
            printf("❌ 单次推理演示失败，错误码: %d\n", single_inference_result);
        }
    }
    
    // 清理资源
    free(results);
    mnist_free_test_data(&test_data);
    inference_destroy(inference_handle);
    
    printf("\n✅ %s 统一推理库测试完成\n", PLATFORM_NAME);
    
    return 0;
} 