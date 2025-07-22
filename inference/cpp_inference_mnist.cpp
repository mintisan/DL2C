// C++ ONNX推理 - 使用真实MNIST测试数据
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

class CppONNXInferenceMNIST {
private:
    const OrtApi* ort_api;
    OrtEnv* env;
    OrtSession* session;
    OrtMemoryInfo* memory_info;
    
public:
    struct InferenceResult {
        int sample_id;
        int original_mnist_index;
        int true_label;
        int predicted_class;
        float confidence;
        std::vector<float> probabilities;
        double inference_time_ms;
        bool is_correct;
    };
    
    CppONNXInferenceMNIST(const std::string& model_path) {
        std::cout << "=== C++ ONNX推理测试 (真实MNIST数据) ===" << std::endl;
        std::cout << "初始化ONNX Runtime C API..." << std::endl;
        
        // 获取API
        ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        
        // 创建环境
        OrtStatus* status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "CppONNXInferenceMNIST", &env);
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
    
    ~CppONNXInferenceMNIST() {
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
    
    InferenceResult inference(int sample_id, int original_idx, int true_label, const std::vector<float>& input_data) {
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
        result.original_mnist_index = original_idx;
        result.true_label = true_label;
        result.predicted_class = predicted_class;
        result.confidence = confidence;
        result.probabilities = probabilities;
        result.inference_time_ms = duration.count() / 1000.0;
        result.is_correct = (predicted_class == true_label);
        
        return result;
    }
};

// 结构体用于存储MNIST测试数据
struct MNISTTestData {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
    std::vector<int> original_indices;
};

// 读取MNIST测试数据
MNISTTestData load_mnist_test_data() {
    std::cout << "🔍 加载MNIST测试数据..." << std::endl;
    
    MNISTTestData data;
    
    // 读取元数据以获取标签和索引信息
    std::ifstream metadata_file("../../test_data_mnist/metadata.json");
    if (!metadata_file.is_open()) {
        throw std::runtime_error("❌ 无法打开元数据文件");
    }
    
    // 简单解析JSON获取信息
    std::string line;
    std::vector<int> labels;
    std::vector<int> indices;
    int num_samples = 0;
    
    while (std::getline(metadata_file, line)) {
        // 获取样本数量
        size_t pos = line.find("\"num_samples\":");
        if (pos != std::string::npos) {
            pos += 14;
            while (pos < line.length() && (line[pos] == ' ' || line[pos] == '"')) pos++;
            size_t end_pos = line.find(',', pos);
            if (end_pos == std::string::npos) end_pos = line.find('}', pos);
            num_samples = std::stoi(line.substr(pos, end_pos - pos));
        }
        
        // 获取标签
        pos = line.find("\"true_label\":");
        if (pos != std::string::npos) {
            pos += 13;
            while (pos < line.length() && (line[pos] == ' ' || line[pos] == '"')) pos++;
            size_t end_pos = line.find(',', pos);
            int label = std::stoi(line.substr(pos, end_pos - pos));
            labels.push_back(label);
        }
        
        // 获取原始索引
        pos = line.find("\"original_mnist_index\":");
        if (pos != std::string::npos) {
            pos += 23;
            while (pos < line.length() && (line[pos] == ' ' || line[pos] == '"')) pos++;
            size_t end_pos = line.find(',', pos);
            int index = std::stoi(line.substr(pos, end_pos - pos));
            indices.push_back(index);
        }
    }
    metadata_file.close();
    
    std::cout << "样本数量: " << num_samples << std::endl;
    std::cout << "解析到的标签数: " << labels.size() << std::endl;
    
    // 读取图像文件
    for (int i = 0; i < num_samples; ++i) {
        std::string filename = std::string("../../test_data_mnist/image_") + 
                              (i < 100 ? (i < 10 ? "00" : "0") : "") + std::to_string(i) + ".bin";
        
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "❌ 无法打开文件: " << filename << std::endl;
            continue;
        }
        
        // 读取28*28个float32值
        std::vector<float> image_data(28 * 28);
        file.read(reinterpret_cast<char*>(image_data.data()), 28 * 28 * sizeof(float));
        file.close();
        
        data.images.push_back(image_data);
        data.labels.push_back(labels[i]);
        data.original_indices.push_back(indices[i]);
    }
    
    std::cout << "✅ 加载了 " << data.images.size() << " 个测试样本" << std::endl;
    
    // 显示标签分布
    std::vector<int> label_count(10, 0);
    for (int label : data.labels) {
        label_count[label]++;
    }
    std::cout << "标签分布: [";
    for (size_t i = 0; i < label_count.size(); ++i) {
        std::cout << label_count[i];
        if (i < label_count.size() - 1) std::cout << " ";
    }
    std::cout << "]" << std::endl;
    
    return data;
}

// 保存结果到JSON文件
void save_results_to_json(const std::vector<CppONNXInferenceMNIST::InferenceResult>& results, 
                         double avg_time, double accuracy, int wrong_count) {
    // 创建目录
    system("mkdir -p ../../results 2>/dev/null || mkdir ..\\..\\results 2>nul || true");
    
    std::ofstream file("../../results/cpp_inference_mnist_results.json");
    if (!file.is_open()) {
        std::cout << "无法创建结果文件" << std::endl;
        return;
    }
    
    file << "{\n";
    file << "  \"platform\": \"C++\",\n";
    file << "  \"framework\": \"ONNX Runtime C++ API\",\n";
    file << "  \"test_type\": \"real_mnist_data\",\n";
    file << "  \"data_source\": \"MNIST test set subset\",\n";
    file << "  \"summary\": {\n";
    file << "    \"accuracy\": " << std::fixed << std::setprecision(4) << accuracy << ",\n";
    file << "    \"average_inference_time_ms\": " << std::setprecision(2) << avg_time << ",\n";
    file << "    \"fps\": " << std::setprecision(1) << (1000.0 / avg_time) << ",\n";
    file << "    \"total_samples\": " << results.size() << ",\n";
    
    int correct = 0;
    for (const auto& r : results) {
        if (r.is_correct) correct++;
    }
    file << "    \"correct_predictions\": " << correct << ",\n";
    file << "    \"wrong_predictions\": " << wrong_count << "\n";
    file << "  },\n";
    file << "  \"results\": [\n";
    
    for (size_t i = 0; i < results.size(); ++i) {
        file << "    {\n";
        file << "      \"sample_id\": " << results[i].sample_id << ",\n";
        file << "      \"original_mnist_index\": " << results[i].original_mnist_index << ",\n";
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
    
    std::cout << "结果已保存到: ../../results/cpp_inference_mnist_results.json" << std::endl;
}

int main() {
    try {
        // 初始化推理引擎
        CppONNXInferenceMNIST inference_engine("../../models/mnist_model.onnx");
        
        // 加载测试数据
        auto test_data = load_mnist_test_data();
        
        if (test_data.images.empty()) {
            std::cout << "❌ 没有加载到测试数据" << std::endl;
            return -1;
        }
        
        std::cout << "\n开始推理 " << test_data.images.size() << " 个样本..." << std::endl;
        
        // 执行推理
        std::vector<CppONNXInferenceMNIST::InferenceResult> results;
        double total_time = 0.0;
        int correct_predictions = 0;
        
        for (size_t i = 0; i < test_data.images.size(); ++i) {
            auto result = inference_engine.inference(
                i, 
                test_data.original_indices[i], 
                test_data.labels[i], 
                test_data.images[i]
            );
            results.push_back(result);
            
            total_time += result.inference_time_ms;
            if (result.is_correct) {
                correct_predictions++;
            }
            
            // 显示进度（每10个样本）
            if ((i + 1) % 10 == 0) {
                double current_accuracy = (double)correct_predictions / (i + 1) * 100;
                std::cout << "完成 " << std::setw(3) << (i+1) << "/" << test_data.images.size() 
                          << " 样本，当前准确率: " << std::fixed << std::setprecision(1) 
                          << current_accuracy << "%" << std::endl;
            }
        }
        
        // 计算统计信息
        double avg_time = total_time / test_data.images.size();
        double accuracy = static_cast<double>(correct_predictions) / test_data.images.size();
        int wrong_count = test_data.images.size() - correct_predictions;
        
        std::cout << "\n=== 推理结果统计 ===" << std::endl;
        std::cout << "总样本数: " << test_data.images.size() << std::endl;
        std::cout << "正确预测: " << correct_predictions << std::endl;
        std::cout << "准确率: " << std::fixed << std::setprecision(2) << accuracy * 100 << "%" << std::endl;
        std::cout << "平均推理时间: " << std::setprecision(2) << avg_time << "ms" << std::endl;
        std::cout << "推理速度: " << std::setprecision(1) << 1000.0 / avg_time << " FPS" << std::endl;
        
        // 显示错误样本
        if (wrong_count > 0) {
            std::cout << "\n❌ 错误预测样本 (" << wrong_count << " 个):" << std::endl;
            int shown = 0;
            for (const auto& result : results) {
                if (!result.is_correct && shown < 5) {
                    std::cout << "  样本 " << std::setw(3) << result.sample_id 
                              << ": 真实=" << result.true_label
                              << ", 预测=" << result.predicted_class
                              << ", 置信度=" << std::setprecision(3) << result.confidence << std::endl;
                    shown++;
                }
            }
            if (wrong_count > 5) {
                std::cout << "  ... 还有 " << (wrong_count - 5) << " 个错误样本" << std::endl;
            }
        }
        
        // 保存结果
        save_results_to_json(results, avg_time, accuracy, wrong_count);
        
        std::cout << "\n✅ C++推理测试完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
} 