#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <memory>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <string>
#include "onnxruntime_c_api.h"

// 平台特定的路径配置
#ifdef __ANDROID__
    #define MODEL_PATH "/data/local/tmp/mnist_unified_onnx/models/mnist_model.onnx"
    #define RESULTS_PATH "/data/local/tmp/mnist_unified_onnx/results/android_unified_cpp_results.txt"
    #define TEST_DATA_DIR "/data/local/tmp/mnist_unified_onnx/test_data_mnist"
    #define PLATFORM_NAME "Android"
#else
    #define MODEL_PATH "../models/mnist_model.onnx"
    #define RESULTS_PATH "../results/macos_unified_cpp_results.txt"
    #define TEST_DATA_DIR "../test_data_mnist"
    #define PLATFORM_NAME "macOS"
#endif

class UnifiedONNXInference {
private:
    const OrtApi* ort_api;
    OrtEnv* env;
    OrtSession* session;
    OrtSessionOptions* session_options;
    OrtMemoryInfo* memory_info;
    
    std::vector<const char*> input_names = {"input"};
    std::vector<const char*> output_names = {"output"};
    
    bool model_loaded = false;
    std::string model_path;

public:
    UnifiedONNXInference() : ort_api(nullptr), env(nullptr), session(nullptr), 
                            session_options(nullptr), memory_info(nullptr) {
        std::cout << "=== " << PLATFORM_NAME << " 统一 C++ ONNX推理测试 ===" << std::endl;
        std::cout << "使用真实MNIST数据进行推理" << std::endl;
    }

    ~UnifiedONNXInference() {
        cleanup();
    }

    bool initialize() {
        model_path = MODEL_PATH;
        
        std::cout << "初始化ONNX Runtime C API..." << std::endl;
        
        // 获取 ONNX Runtime API
        ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (!ort_api) {
            std::cerr << "错误: 无法获取 ONNX Runtime API" << std::endl;
            return false;
        }

        // 创建环境
        OrtStatus* status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "UnifiedInference", &env);
        if (status != nullptr) {
            std::cerr << "错误: 创建环境失败: " << ort_api->GetErrorMessage(status) << std::endl;
            ort_api->ReleaseStatus(status);
            return false;
        }

        // 创建会话选项
        status = ort_api->CreateSessionOptions(&session_options);
        if (status != nullptr) {
            std::cerr << "错误: 创建会话选项失败: " << ort_api->GetErrorMessage(status) << std::endl;
            ort_api->ReleaseStatus(status);
            return false;
        }

        // 设置线程数
        status = ort_api->SetIntraOpNumThreads(session_options, 1);
        if (status != nullptr) {
            ort_api->ReleaseStatus(status);
        }

        // 创建内存信息
        status = ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
        if (status != nullptr) {
            std::cerr << "错误: 创建内存信息失败: " << ort_api->GetErrorMessage(status) << std::endl;
            ort_api->ReleaseStatus(status);
            return false;
        }

        // 加载模型
        status = ort_api->CreateSession(env, model_path.c_str(), session_options, &session);
        if (status != nullptr) {
            std::cerr << "❌ 推理测试失败: 加载模型失败: " << model_path << std::endl;
            ort_api->ReleaseStatus(status);
            return false;
        }

        model_loaded = true;
        std::cout << "✅ 模型加载成功: " << model_path << std::endl;
        
        return true;
    }

    void cleanup() {
        if (session) {
            ort_api->ReleaseSession(session);
            session = nullptr;
        }
        if (memory_info) {
            ort_api->ReleaseMemoryInfo(memory_info);
            memory_info = nullptr;
        }
        if (session_options) {
            ort_api->ReleaseSessionOptions(session_options);
            session_options = nullptr;
        }
        if (env) {
            ort_api->ReleaseEnv(env);
            env = nullptr;
        }
    }

    // 从metadata.json加载标签映射（与原始android_real_onnx_inference.cpp一致）
    std::map<int, int> loadLabelsFromMetadata() {
        std::map<int, int> image_to_label;
        
        std::string metadata_path = std::string(TEST_DATA_DIR) + "/metadata.json";
        std::ifstream metadata_file(metadata_path);
        if (!metadata_file.is_open()) {
            std::cerr << "❌ 无法打开元数据文件: " << metadata_path << std::endl;
            std::cout << "使用默认标签映射..." << std::endl;
            // 提供前10个样本的后备标签
            image_to_label[0] = 2; image_to_label[1] = 1; image_to_label[2] = 1; 
            image_to_label[3] = 1; image_to_label[4] = 2; image_to_label[5] = 6;
            image_to_label[6] = 3; image_to_label[7] = 8; image_to_label[8] = 2; 
            image_to_label[9] = 6;
            return image_to_label;
        }
        
        // 简单解析JSON获取标签信息
        std::string line;
        std::vector<int> labels;
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
        }
        metadata_file.close();
        
        // 构建样本索引到标签的映射
        for (int i = 0; i < std::min(num_samples, (int)labels.size()); ++i) {
            image_to_label[i] = labels[i];
        }
        
        std::cout << "✓ 已从metadata.json加载 " << image_to_label.size() << " 个样本的标签信息" << std::endl;
        return image_to_label;
    }

    // 加载真实测试数据（与原始版本逻辑一致）
    std::vector<float> loadTestData(const std::string& filename, int& expected_label) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "警告: 无法打开测试数据文件: " << filename << std::endl;
            std::cout << "使用随机测试数据..." << std::endl;
            expected_label = 7; // 随机标签
            
            // 生成随机测试数据
            std::vector<float> data(784);
            for (int i = 0; i < 784; i++) {
                data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
            return data;
        }

        // 读取图像数据（与原始版本一致）
        std::vector<float> data(784);
        file.read(reinterpret_cast<char*>(data.data()), 784 * sizeof(float));
        file.close();

        std::cout << "✓ 加载测试数据: " << filename << std::endl;
        return data;
    }

    std::pair<int, double> runInference(const std::vector<float>& input_data) {
        if (!model_loaded) {
            std::cerr << "错误: 模型未加载" << std::endl;
            return {-1, 0.0};
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // 添加预处理步骤（与原始版本保持一致）
        auto processed_data = preprocess(input_data);

        // 创建输入张量
        std::vector<int64_t> input_shape = {1, 1, 28, 28};
        OrtValue* input_tensor = nullptr;
        
        OrtStatus* status = ort_api->CreateTensorWithDataAsOrtValue(
            memory_info,
            const_cast<float*>(processed_data.data()),
            processed_data.size() * sizeof(float),
            input_shape.data(),
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor
        );

        if (status != nullptr) {
            std::cerr << "错误: 创建输入张量失败: " << ort_api->GetErrorMessage(status) << std::endl;
            ort_api->ReleaseStatus(status);
            return {-1, 0.0};
        }

        // 运行推理
        OrtValue* output_tensor = nullptr;
        status = ort_api->Run(
            session,
            nullptr,  // RunOptions
            input_names.data(),
            (const OrtValue* const*)&input_tensor,
            1,  // 输入数量
            output_names.data(),
            1,  // 输出数量
            &output_tensor
        );

        if (status != nullptr) {
            std::cerr << "错误: 推理执行失败: " << ort_api->GetErrorMessage(status) << std::endl;
            ort_api->ReleaseStatus(status);
            ort_api->ReleaseValue(input_tensor);
            return {-1, 0.0};
        }

        // 获取输出数据
        float* output_data;
        status = ort_api->GetTensorMutableData(output_tensor, (void**)&output_data);
        if (status != nullptr) {
            std::cerr << "错误: 获取输出数据失败: " << ort_api->GetErrorMessage(status) << std::endl;
            ort_api->ReleaseStatus(status);
            ort_api->ReleaseValue(input_tensor);
            ort_api->ReleaseValue(output_tensor);
            return {-1, 0.0};
        }

        // 复制输出数据并应用softmax（与原始版本保持一致）
        std::vector<float> logits(output_data, output_data + 10);  // MNIST有10个类别
        std::vector<float> probabilities = softmax(logits);
        
        // 找到预测类别
        auto max_it = std::max_element(probabilities.begin(), probabilities.end());
        int predicted_class = std::distance(probabilities.begin(), max_it);
        float max_prob = *max_it;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double inference_time_ms = duration.count() / 1000.0;

        // 清理资源
        ort_api->ReleaseValue(input_tensor);
        ort_api->ReleaseValue(output_tensor);

        return {predicted_class, inference_time_ms};
    }

    void runTests() {
        if (!model_loaded) {
            std::cerr << "错误: 模型未加载，无法运行测试" << std::endl;
            return;
        }

        std::cout << "加载" << PLATFORM_NAME << "测试数据..." << std::endl;

        // 加载正确的标签信息
        auto label_map = loadLabelsFromMetadata();
        
        std::vector<std::string> test_files;
        std::vector<std::pair<int, double>> results;
        int correct_predictions = 0;
        double total_time = 0.0;

        // 测试所有样本（与原始版本完全一致）
        int num_samples = label_map.size();
        std::cout << "✓ 加载 " << num_samples << " 个" << PLATFORM_NAME << "测试样本" << std::endl;
        std::cout << "\n=== 开始 " << PLATFORM_NAME << " 统一推理测试 ===" << std::endl;
        std::cout << "开始推理 " << num_samples << " 个样本..." << std::endl;
        
        for (int idx = 0; idx < num_samples; ++idx) {
            std::string filename = std::string(TEST_DATA_DIR) + "/image_" + 
                                 std::string(3 - std::to_string(idx).length(), '0') + 
                                 std::to_string(idx) + ".bin";
            int expected_label;
            auto input_data = loadTestData(filename, expected_label);
            
            // 使用正确的标签
            if (label_map.find(idx) != label_map.end()) {
                expected_label = label_map[idx];
            } else {
                std::cerr << "警告: 找不到样本 " << idx << " 的标签信息" << std::endl;
                continue;
            }
            
            auto result = runInference(input_data);
            int predicted_class = result.first;
            double inference_time = result.second;
            
            if (predicted_class >= 0) {
                results.push_back(result);
                total_time += inference_time;
                
                bool correct = (predicted_class == expected_label);
                if (correct) correct_predictions++;
                
                // 显示进度（每10个样本）
                if ((idx + 1) % 10 == 0) {
                    double current_accuracy = (double)correct_predictions / (idx + 1) * 100;
                    std::cout << "完成 " << std::setw(3) << (idx+1) << "/" << num_samples 
                              << " 样本，当前准确率: " << std::fixed << std::setprecision(1) 
                              << current_accuracy << "%" << std::endl;
                }
            }
        }

        if (!results.empty()) {
            double avg_time = total_time / results.size();
            double accuracy = static_cast<double>(correct_predictions) / results.size() * 100.0;
            double fps = 1000.0 / avg_time;
            int wrong_count = results.size() - correct_predictions;

            std::cout << "\n=== " << PLATFORM_NAME << " 推理结果统计 ===" << std::endl;
            std::cout << "总样本数: " << results.size() << std::endl;
            std::cout << "正确预测: " << correct_predictions << std::endl;
            std::cout << "准确率: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
            std::cout << "平均推理时间: " << std::fixed << std::setprecision(2) << avg_time << " ms" << std::endl;
            std::cout << "推理速度: " << std::fixed << std::setprecision(1) << fps << " FPS" << std::endl;
            
            // 显示错误样本（与原始版本保持一致）
            if (wrong_count > 0) {
                std::cout << "\n❌ 错误预测样本 (" << wrong_count << " 个):" << std::endl;
                int shown = 0;
                for (int idx = 0; idx < results.size() && shown < 5; ++idx) {
                    if (label_map.find(idx) != label_map.end()) {
                        int expected_label = label_map[idx];
                        int predicted_class = results[idx].first;
                        if (predicted_class != expected_label) {
                            double inference_time = results[idx].second;
                            std::cout << "  样本 " << std::setw(3) << idx 
                                      << ": 真实=" << expected_label
                                      << ", 预测=" << predicted_class
                                      << ", 时间=" << std::setprecision(2) << inference_time << " ms" << std::endl;
                            shown++;
                        }
                    }
                }
                if (wrong_count > 5) {
                    std::cout << "  ... 还有 " << (wrong_count - 5) << " 个错误样本" << std::endl;
                }
            }
            
            // 保存结果到文件
            saveResults(results, label_map, accuracy, avg_time, fps);
        } else {
            std::cout << "没有成功的推理结果" << std::endl;
        }
        
        std::cout << "\n✅ " << PLATFORM_NAME << " 统一推理测试完成" << std::endl;
    }

private:
    // 添加预处理函数（与原始版本保持一致）
    std::vector<float> preprocess(const std::vector<float>& raw_data) {
        std::vector<float> processed = raw_data;
        
        // MNIST标准化参数（与原始版本保持一致）
        const float mean = 0.1307f;
        const float std = 0.3081f;
        
        // 标准化: (pixel - mean) / std
        for (auto& pixel : processed) {
            pixel = (pixel - mean) / std;
        }
        
        return processed;
    }

    // 添加softmax函数（与原始版本保持一致）
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

    void saveResults(const std::vector<std::pair<int, double>>& results, 
                    const std::map<int, int>& label_map,
                    double accuracy, double avg_time, double fps) {
        std::ofstream file(RESULTS_PATH);
        if (file.is_open()) {
            file << PLATFORM_NAME << " 统一 ONNX Runtime C++ 推理结果\n";
            file << "==========================================\n";
            file << "平台: " << PLATFORM_NAME << "\n";
            file << "总样本数: " << results.size() << "\n";
            file << "准确率: " << std::fixed << std::setprecision(2) << accuracy << "%\n";
            file << "平均推理时间: " << std::fixed << std::setprecision(2) << avg_time << " ms\n";
            file << "推理速度: " << std::fixed << std::setprecision(1) << fps << " FPS\n\n";
            
            file << "样本详细结果:\n";
            for (int idx = 0; idx < results.size(); ++idx) {
                if (label_map.find(idx) != label_map.end()) {
                    int expected_label = label_map.at(idx);
                    int predicted_class = results[idx].first;
                    double inference_time = results[idx].second;
                    bool correct = (predicted_class == expected_label);
                    
                    file << "样本 " << std::setw(3) << idx 
                         << ": 真实=" << expected_label
                         << ", 预测=" << predicted_class
                         << ", 置信度=" << std::setprecision(3) << "N/A"  // 在此版本中不保存置信度
                         << ", 时间=" << std::setprecision(2) << inference_time << " ms, "
                         << (correct ? "正确" : "错误") << "\n";
                }
            }
            
            file.close();
            std::cout << "✓ 结果已保存到 " << RESULTS_PATH << std::endl;
        }
    }
};

int main() {
    std::cout << "启动 " << PLATFORM_NAME << " 统一 C++ ONNX推理程序..." << std::endl;
    
    UnifiedONNXInference inference;
    
    // 初始化
    if (!inference.initialize()) {
        std::cerr << "初始化失败" << std::endl;
        return -1;
    }
    
    // 运行测试
    inference.runTests();
    
    std::cout << "\n" << PLATFORM_NAME << " 统一推理测试完成！" << std::endl;
    return 0;
} 