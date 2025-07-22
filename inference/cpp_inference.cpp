// 简化的C++ ONNX推理实现，避免float16依赖
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>
#include <stdexcept>
#include <fstream>

// 直接使用C API避免C++头文件依赖问题
extern "C" {
#include "onnxruntime_c_api.h"
}

class SimpleCppONNXInference {
private:
    const OrtApi* ort_api;
    OrtEnv* env;
    OrtSession* session;
    OrtMemoryInfo* memory_info;
    
public:
    SimpleCppONNXInference(const std::string& model_path) {
        std::cout << "=== C++ ONNX推理测试 (简化版) ===" << std::endl;
        std::cout << "初始化ONNX Runtime C API..." << std::endl;
        
        // 获取API
        ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        
        // 创建环境
        OrtStatus* status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "SimpleCppONNXInference", &env);
        if (status != nullptr) {
            ort_api->ReleaseStatus(status);
            throw std::runtime_error("创建环境失败");
        }
        
        // 创建会话选项
        OrtSessionOptions* session_options;
        status = ort_api->CreateSessionOptions(&session_options);
        if (status != nullptr) {
            ort_api->ReleaseStatus(status);
            throw std::runtime_error("创建会话选项失败");
        }
        
        // 创建会话
        status = ort_api->CreateSession(env, model_path.c_str(), session_options, &session);
        if (status != nullptr) {
            ort_api->ReleaseSessionOptions(session_options);
            ort_api->ReleaseStatus(status);
            throw std::runtime_error("加载模型失败: " + model_path);
        }
        
        // 创建内存信息
        status = ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
        if (status != nullptr) {
            ort_api->ReleaseStatus(status);
            throw std::runtime_error("创建内存信息失败");
        }
        
        ort_api->ReleaseSessionOptions(session_options);
        std::cout << "✅ 模型加载成功: " << model_path << std::endl;
    }
    
    ~SimpleCppONNXInference() {
        if (memory_info) ort_api->ReleaseMemoryInfo(memory_info);
        if (session) ort_api->ReleaseSession(session);
        if (env) ort_api->ReleaseEnv(env);
    }
    
    struct InferenceResult {
        int predicted_class;
        float confidence;
        std::vector<float> probabilities;
        double inference_time_ms;
    };
    
    std::vector<float> preprocess(const std::vector<float>& raw_data) {
        std::vector<float> processed = raw_data;
        
        // MNIST标准化
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
        int64_t input_shape[] = {1, 1, 28, 28};
        OrtValue* input_tensor = nullptr;
        
        OrtStatus* status = ort_api->CreateTensorWithDataAsOrtValue(
            memory_info,
            processed_data.data(),
            processed_data.size() * sizeof(float),
            input_shape,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor
        );
        
        if (status != nullptr) {
            ort_api->ReleaseStatus(status);
            throw std::runtime_error("创建输入tensor失败");
        }
        
        // 设置输入输出名称
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};
        
        // 运行推理
        OrtValue* output_tensor = nullptr;
        status = ort_api->Run(
            session,
            nullptr,
            input_names,
            (const OrtValue* const*)&input_tensor,
            1,
            output_names,
            1,
            &output_tensor
        );
        
        if (status != nullptr) {
            ort_api->ReleaseValue(input_tensor);
            ort_api->ReleaseStatus(status);
            throw std::runtime_error("推理执行失败");
        }
        
        // 获取输出数据
        float* output_data;
        status = ort_api->GetTensorMutableData(output_tensor, (void**)&output_data);
        if (status != nullptr) {
            ort_api->ReleaseValue(input_tensor);
            ort_api->ReleaseValue(output_tensor);
            ort_api->ReleaseStatus(status);
            throw std::runtime_error("获取输出数据失败");
        }
        
        // 获取输出形状
        OrtTensorTypeAndShapeInfo* output_info;
        status = ort_api->GetTensorTypeAndShape(output_tensor, &output_info);
        if (status != nullptr) {
            ort_api->ReleaseValue(input_tensor);
            ort_api->ReleaseValue(output_tensor);
            ort_api->ReleaseStatus(status);
            throw std::runtime_error("获取输出形状失败");
        }
        
        size_t output_count;
        status = ort_api->GetTensorShapeElementCount(output_info, &output_count);
        if (status != nullptr) {
            ort_api->ReleaseValue(input_tensor);
            ort_api->ReleaseValue(output_tensor);
            ort_api->ReleaseTensorTypeAndShapeInfo(output_info);
            ort_api->ReleaseStatus(status);
            throw std::runtime_error("获取输出元素数量失败");
        }
        
        // 复制输出数据
        std::vector<float> logits(output_data, output_data + output_count);
        std::vector<float> probabilities = softmax(logits);
        
        // 找到预测类别
        auto max_it = std::max_element(probabilities.begin(), probabilities.end());
        int predicted_class = std::distance(probabilities.begin(), max_it);
        float confidence = *max_it;
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // 清理资源
        ort_api->ReleaseValue(input_tensor);
        ort_api->ReleaseValue(output_tensor);
        ort_api->ReleaseTensorTypeAndShapeInfo(output_info);
        
        InferenceResult result;
        result.predicted_class = predicted_class;
        result.confidence = confidence;
        result.probabilities = probabilities;
        result.inference_time_ms = duration.count() / 1000.0;
        
        return result;
    }
};

// 生成测试数据
std::vector<float> generate_test_image(int label) {
    std::vector<float> image(28 * 28, 0.0f);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    // 生成简单的模拟数据
    for (int i = 0; i < 28 * 28; ++i) {
        image[i] = dis(gen);
    }
    
    return image;
}

// 保存结果到JSON文件
void save_results_to_json(const std::vector<SimpleCppONNXInference::InferenceResult>& results, double avg_time) {
    // 尝试创建results目录
    system("mkdir -p ../../results 2>/dev/null || mkdir ..\\..\\results 2>nul || true");
    
    std::ofstream file("../../results/cpp_inference_results.json");
    if (!file.is_open()) {
        std::cout << "无法创建结果文件" << std::endl;
        return;
    }
    
    file << "{\n";
    file << "  \"platform\": \"C++\",\n";
    file << "  \"framework\": \"ONNX Runtime C++ API\",\n";
    file << "  \"summary\": {\n";
    file << "    \"total_samples\": " << results.size() << ",\n";
    file << "    \"average_inference_time_ms\": " << std::fixed << std::setprecision(2) << avg_time << ",\n";
    file << "    \"fps\": " << std::setprecision(1) << (1000.0 / avg_time) << "\n";
    file << "  },\n";
    file << "  \"results\": [\n";
    
    for (size_t i = 0; i < results.size(); ++i) {
        file << "    {\n";
        file << "      \"sample_id\": " << i << ",\n";
        file << "      \"predicted_class\": " << results[i].predicted_class << ",\n";
        file << "      \"confidence\": " << std::setprecision(4) << results[i].confidence << ",\n";
        file << "      \"inference_time_ms\": " << std::setprecision(2) << results[i].inference_time_ms << "\n";
        file << "    }";
        if (i < results.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    file.close();
    
    std::cout << "结果已保存到: ../../results/cpp_inference_results.json" << std::endl;
}

int main() {
    try {
        // 初始化推理引擎
        SimpleCppONNXInference inference_engine("../../models/mnist_model.onnx");
        
        std::cout << "\n开始推理测试..." << std::endl;
        
        // 测试10个样本
        std::vector<SimpleCppONNXInference::InferenceResult> all_results;
        double total_time = 0.0;
        
        for (int i = 0; i < 10; ++i) {
            // 生成测试数据
            auto test_image = generate_test_image(i % 10);
            
            // 执行推理
            auto result = inference_engine.inference(test_image);
            all_results.push_back(result);
            
            // 对于模拟数据，我们无法验证准确性，只测试推理是否工作
            std::cout << "样本 " << i 
                      << ": 预测=" << result.predicted_class
                      << ", 置信度=" << std::fixed << std::setprecision(4) << result.confidence
                      << ", 时间=" << std::setprecision(2) << result.inference_time_ms << "ms"
                      << std::endl;
            
            total_time += result.inference_time_ms;
        }
        
        double avg_time = total_time / 10;
        std::cout << std::endl;
        std::cout << "平均推理时间: " << std::setprecision(2) << avg_time << "ms" << std::endl;
        std::cout << "推理速度: " << std::setprecision(1) << 1000.0 / avg_time << " FPS" << std::endl;
        
        // 保存结果到JSON文件
        save_results_to_json(all_results, avg_time);
        
        std::cout << "\n✅ C++推理测试完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 