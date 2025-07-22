// C++ ONNX推理 - 使用共同测试数据
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

// 使用C API避免依赖问题
extern "C" {
#include "onnxruntime_c_api.h"
}

// 前向声明
class CppONNXInferenceCommon;

// JSON简单输出类
class SimpleJSON {
public:
    template<typename ResultType>
    static void writeInferenceResults(const std::string& filename, 
                                    const std::vector<ResultType>& results,
                                    double avg_time, double accuracy) {
        std::ofstream file(filename);
        if (!file.is_open()) return;
        
        file << "{\n";
        file << "  \"platform\": \"C++\",\n";
        file << "  \"framework\": \"ONNX Runtime C++ API\",\n";
        file << "  \"test_type\": \"common_data\",\n";
        file << "  \"summary\": {\n";
        file << "    \"accuracy\": " << std::fixed << std::setprecision(4) << accuracy << ",\n";
        file << "    \"average_inference_time_ms\": " << std::setprecision(2) << avg_time << ",\n";
        file << "    \"fps\": " << std::setprecision(1) << (1000.0 / avg_time) << ",\n";
        file << "    \"total_samples\": " << results.size() << ",\n";
        
        int correct = 0;
        for (const auto& r : results) {
            if (r.is_correct) correct++;
        }
        file << "    \"correct_predictions\": " << correct << "\n";
        file << "  },\n";
        file << "  \"results\": [\n";
        
        for (size_t i = 0; i < results.size(); ++i) {
            file << "    {\n";
            file << "      \"sample_id\": " << i << ",\n";
            file << "      \"true_label\": " << results[i].true_label << ",\n";
            file << "      \"predicted_class\": " << results[i].predicted_class << ",\n";
            file << "      \"confidence\": " << std::setprecision(4) << results[i].confidence << ",\n";
            file << "      \"inference_time_ms\": " << std::setprecision(2) << results[i].inference_time_ms << ",\n";
            file << "      \"is_correct\": " << (results[i].is_correct ? "true" : "false") << "\n";
            file << "    }";
            if (i < results.size() - 1) file << ",";
            file << "\n";
        }
        
        file << "  ]\n";
        file << "}\n";
        file.close();
    }
};

class CppONNXInferenceCommon {
private:
    const OrtApi* ort_api;
    OrtEnv* env;
    OrtSession* session;
    OrtMemoryInfo* memory_info;
    
public:
    struct InferenceResult {
        int sample_id;
        int true_label;
        int predicted_class;
        float confidence;
        std::vector<float> probabilities;
        double inference_time_ms;
        bool is_correct;
    };
    
    CppONNXInferenceCommon(const std::string& model_path) {
        std::cout << "=== C++ ONNX推理测试 (共同数据) ===" << std::endl;
        std::cout << "初始化ONNX Runtime C API..." << std::endl;
        
        // 获取API
        ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        
        // 创建环境
        OrtStatus* status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "CppONNXInferenceCommon", &env);
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
    
    ~CppONNXInferenceCommon() {
        if (memory_info) ort_api->ReleaseMemoryInfo(memory_info);
        if (session) ort_api->ReleaseSession(session);
        if (env) ort_api->ReleaseEnv(env);
    }
    
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
    
    InferenceResult inference(int sample_id, int true_label, const std::vector<float>& input_data) {
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
        result.sample_id = sample_id;
        result.true_label = true_label;
        result.predicted_class = predicted_class;
        result.confidence = confidence;
        result.probabilities = probabilities;
        result.inference_time_ms = duration.count() / 1000.0;
        result.is_correct = (predicted_class == true_label);
        
        return result;
    }
};

// 读取测试数据
std::vector<std::vector<float>> load_test_images(std::vector<int>& true_labels) {
    std::cout << "🔍 加载共同测试数据..." << std::endl;
    
    std::vector<std::vector<float>> test_images;
    
    // 读取元数据以获取标签信息
    std::ifstream metadata_file("../../test_data/metadata.json");
    if (!metadata_file.is_open()) {
        std::cout << "❌ 无法打开元数据文件" << std::endl;
        return test_images;
    }
    
    // 简单解析JSON获取标签（这里用简单方法，实际项目中应该用JSON库）
    std::string line;
    std::vector<int> labels;
    while (std::getline(metadata_file, line)) {
        size_t pos = line.find("\"true_label\":");
        if (pos != std::string::npos) {
            pos += 13; // 跳过 "true_label":
            while (pos < line.length() && (line[pos] == ' ' || line[pos] == '"')) pos++;
            int label = std::stoi(line.substr(pos, 1));
            labels.push_back(label);
        }
    }
    metadata_file.close();
    
    // 读取二进制图像文件
    for (int i = 0; i < 10; ++i) {
        std::string filename = std::string("../../test_data/sample_") + 
                              (i < 10 ? "0" : "") + std::to_string(i) + ".bin";
        
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "❌ 无法打开文件: " << filename << std::endl;
            continue;
        }
        
        // 读取28*28个float32值
        std::vector<float> image_data(28 * 28);
        file.read(reinterpret_cast<char*>(image_data.data()), 28 * 28 * sizeof(float));
        file.close();
        
        test_images.push_back(image_data);
        true_labels.push_back(labels[i]);
        
        std::cout << "样本 " << i << ": 真实标签=" << labels[i] << std::endl;
    }
    
    std::cout << "✅ 加载了 " << test_images.size() << " 个测试样本" << std::endl;
    return test_images;
}

int main() {
    try {
        // 初始化推理引擎
        CppONNXInferenceCommon inference_engine("../../models/mnist_model.onnx");
        
        // 加载测试数据
        std::vector<int> true_labels;
        auto test_images = load_test_images(true_labels);
        
        if (test_images.empty()) {
            std::cout << "❌ 没有加载到测试数据" << std::endl;
            return -1;
        }
        
        std::cout << "\n开始推理 " << test_images.size() << " 个样本..." << std::endl;
        
        // 执行推理
        std::vector<CppONNXInferenceCommon::InferenceResult> results;
        double total_time = 0.0;
        int correct_predictions = 0;
        
        for (size_t i = 0; i < test_images.size(); ++i) {
            auto result = inference_engine.inference(i, true_labels[i], test_images[i]);
            results.push_back(result);
            
            total_time += result.inference_time_ms;
            if (result.is_correct) {
                correct_predictions++;
            }
            
            std::cout << "样本 " << std::setw(2) << i 
                      << ": 真实=" << result.true_label
                      << ", 预测=" << result.predicted_class
                      << ", 置信度=" << std::fixed << std::setprecision(4) << result.confidence
                      << ", 时间=" << std::setprecision(2) << result.inference_time_ms << "ms"
                      << ", " << (result.is_correct ? "✓" : "✗")
                      << std::endl;
        }
        
        // 计算统计信息
        double avg_time = total_time / test_images.size();
        double accuracy = static_cast<double>(correct_predictions) / test_images.size();
        
        std::cout << "\n=== 推理结果统计 ===" << std::endl;
        std::cout << "总样本数: " << test_images.size() << std::endl;
        std::cout << "正确预测: " << correct_predictions << std::endl;
        std::cout << "准确率: " << std::fixed << std::setprecision(2) << accuracy * 100 << "%" << std::endl;
        std::cout << "平均推理时间: " << std::setprecision(2) << avg_time << "ms" << std::endl;
        std::cout << "推理速度: " << std::setprecision(1) << 1000.0 / avg_time << " FPS" << std::endl;
        
        // 保存结果
        system("mkdir -p ../../results 2>/dev/null || mkdir ..\\..\\results 2>nul || true");
        SimpleJSON::writeInferenceResults("../../results/cpp_inference_common_results.json", 
                                        results, avg_time, accuracy);
        std::cout << "结果已保存到: ../../results/cpp_inference_common_results.json" << std::endl;
        
        std::cout << "\n✅ C++推理测试完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 