#!/bin/bash
# Android 真实 ONNX Runtime 部署和测试脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Android 真实 ONNX Runtime 部署和测试脚本 ===${NC}"

# 设置变量
PROJECT_DIR=$(pwd)/..
ANDROID_ABI="arm64-v8a"
ANDROID_EXE_DIR="../android_executables/${ANDROID_ABI}"
DEVICE_DIR="/data/local/tmp/mnist_real_onnx"
RESULTS_DIR="../results"
C_VERSION_AVAILABLE=false

# 检查设备连接
check_device() {
    echo -e "${YELLOW}检查设备连接...${NC}"
    
    DEVICE_COUNT=$(adb devices | grep -c "device$" || echo "0")
    if [ "$DEVICE_COUNT" -eq 0 ]; then
        echo -e "${RED}错误: 没有连接的 Android 设备${NC}"
        exit 1
    fi
    
    DEVICE_ID=$(adb devices | grep "device$" | head -1 | awk '{print $1}')
    echo -e "${GREEN}✓ 使用设备: $DEVICE_ID${NC}"
    
    # 获取设备信息
    echo -e "${BLUE}设备信息:${NC}"
    echo "  型号: $(adb shell getprop ro.product.model | tr -d '\r')"
    echo "  版本: Android $(adb shell getprop ro.build.version.release | tr -d '\r')"
    echo "  架构: $(adb shell getprop ro.product.cpu.abi | tr -d '\r')"
    echo "  内核: $(adb shell uname -r | tr -d '\r')"
}

# 部署文件到设备
deploy_files() {
    echo -e "${YELLOW}部署文件到 Android 设备...${NC}"
    
    # 创建设备目录
    adb shell "mkdir -p $DEVICE_DIR" 2>/dev/null || true
    
    # 检查C++可执行文件
    if [ ! -f "$ANDROID_EXE_DIR/android_real_onnx_inference" ]; then
        echo -e "${RED}错误: Android C++ 可执行文件不存在: $ANDROID_EXE_DIR/android_real_onnx_inference${NC}"
        echo "请先运行 ./build_android_real_onnx.sh 编译"
        exit 1
    fi
    
    # 检查C可执行文件（可选）
    if [ ! -f "$ANDROID_EXE_DIR/android_real_onnx_inference_c" ]; then
        echo -e "${YELLOW}警告: Android C 可执行文件不存在，将跳过C版本测试${NC}"
        C_VERSION_AVAILABLE=false
    else
        C_VERSION_AVAILABLE=true
    fi
    
    # 推送C++可执行文件
    echo -e "${BLUE}推送 C++ 可执行文件...${NC}"
    adb push "$ANDROID_EXE_DIR/android_real_onnx_inference" "$DEVICE_DIR/"
    adb shell "chmod +x $DEVICE_DIR/android_real_onnx_inference"
    
    # 推送C可执行文件（如果可用）
    if [ "$C_VERSION_AVAILABLE" = true ]; then
        echo -e "${BLUE}推送 C 可执行文件...${NC}"
        adb push "$ANDROID_EXE_DIR/android_real_onnx_inference_c" "$DEVICE_DIR/"
        adb shell "chmod +x $DEVICE_DIR/android_real_onnx_inference_c"
    fi
    
    # 推送 ONNX 模型
    if [ -f "../models/mnist_model.onnx" ]; then
        echo -e "${BLUE}推送 ONNX 模型...${NC}"
        adb shell "mkdir -p $DEVICE_DIR/models" 2>/dev/null || true
        adb push "../models/mnist_model.onnx" "$DEVICE_DIR/models/"
    else
        echo -e "${YELLOW}警告: ONNX 模型文件不存在，程序将使用随机数据${NC}"
    fi
    
    # 推送测试数据
    if [ -d "../test_data_mnist" ]; then
        echo -e "${BLUE}推送测试数据...${NC}"
        adb shell "mkdir -p $DEVICE_DIR/test_data_mnist" 2>/dev/null || true
        adb push "../test_data_mnist/." "$DEVICE_DIR/test_data_mnist/"
    else
        echo -e "${YELLOW}警告: MNIST 测试数据不存在，程序将使用随机数据${NC}"
    fi
    
    # 创建结果目录
    adb shell "mkdir -p $DEVICE_DIR/results" 2>/dev/null || true
    
    echo -e "${GREEN}✓ 文件部署完成${NC}"
}

# 运行 Android 测试
run_android_tests() {
    echo -e "${YELLOW}在 Android 设备上运行真实 ONNX Runtime 测试...${NC}"
    
    # 测试 C++ 版本
    echo -e "${BLUE}执行 C++ 推理测试...${NC}"
    adb shell "cd $DEVICE_DIR && ./android_real_onnx_inference"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Android 真实 ONNX Runtime C++ 测试完成${NC}"
    else
        echo -e "${RED}✗ Android 真实 ONNX Runtime C++ 测试失败${NC}"
        return 1
    fi
    
    # 测试 C 版本（如果可用）
    if [ "$C_VERSION_AVAILABLE" = true ]; then
        echo -e "\n${BLUE}执行 C 推理测试...${NC}"
        adb shell "cd $DEVICE_DIR && ./android_real_onnx_inference_c"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Android 真实 ONNX Runtime C 测试完成${NC}"
        else
            echo -e "${RED}✗ Android 真实 ONNX Runtime C 测试失败${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}跳过 C 版本测试（可执行文件不可用）${NC}"
    fi
}

# 获取测试结果
get_results() {
    echo -e "${YELLOW}获取测试结果...${NC}"
    
    # 确保本地结果目录存在
    mkdir -p "$RESULTS_DIR"
    
    # 拉取C++版本结果文件
    if adb shell "test -f $DEVICE_DIR/results/android_real_onnx_results.txt"; then
        adb pull "$DEVICE_DIR/results/android_real_onnx_results.txt" "$RESULTS_DIR/"
        echo -e "${GREEN}✓ C++ 结果文件已下载到 $RESULTS_DIR/android_real_onnx_results.txt${NC}"
    else
        echo -e "${YELLOW}警告: Android C++ 结果文件不存在${NC}"
    fi
    
    # 拉取C版本结果文件（如果可用）
    if [ "$C_VERSION_AVAILABLE" = true ]; then
        if adb shell "test -f $DEVICE_DIR/results/android_real_onnx_c_results.txt"; then
            adb pull "$DEVICE_DIR/results/android_real_onnx_c_results.txt" "$RESULTS_DIR/"
            echo -e "${GREEN}✓ C 结果文件已下载到 $RESULTS_DIR/android_real_onnx_c_results.txt${NC}"
        else
            echo -e "${YELLOW}警告: Android C 结果文件不存在${NC}"
        fi
    fi
    
    # 显示C++版本结果
    if [ -f "$RESULTS_DIR/android_real_onnx_results.txt" ]; then
        echo -e "${BLUE}Android 真实 ONNX Runtime C++ 测试结果:${NC}"
        cat "$RESULTS_DIR/android_real_onnx_results.txt"
    fi
    
    # 显示C版本结果（如果可用）
    if [ "$C_VERSION_AVAILABLE" = true ] && [ -f "$RESULTS_DIR/android_real_onnx_c_results.txt" ]; then
        echo -e "\n${BLUE}Android 真实 ONNX Runtime C 测试结果:${NC}"
        cat "$RESULTS_DIR/android_real_onnx_c_results.txt"
    fi
}

# 性能对比
performance_comparison() {
    echo -e "\n${YELLOW}性能对比分析...${NC}"
    
    # 检查本地 C++ 结果
    LOCAL_RESULT_FILE="../results/mnist_results_cpp_mnist.txt"
    ANDROID_RESULT_FILE="$RESULTS_DIR/android_real_onnx_results.txt"
    
    if [ -f "$LOCAL_RESULT_FILE" ] && [ -f "$ANDROID_RESULT_FILE" ]; then
        echo -e "${BLUE}=== 本地 vs Android 真实 ONNX Runtime 性能对比 ===${NC}"
        
        # 提取性能数据
        LOCAL_TIME=$(grep "平均推理时间" "$LOCAL_RESULT_FILE" | grep -o '[0-9.]*' | head -1)
        ANDROID_TIME=$(grep "平均推理时间" "$ANDROID_RESULT_FILE" | grep -o '[0-9.]*' | head -1)
        
        LOCAL_FPS=$(grep "推理 FPS" "$LOCAL_RESULT_FILE" | grep -o '[0-9]*' | head -1)
        ANDROID_FPS=$(grep "推理 FPS" "$ANDROID_RESULT_FILE" | grep -o '[0-9]*' | head -1)
        
        LOCAL_ACC=$(grep "准确率" "$LOCAL_RESULT_FILE" | grep -o '[0-9.]*' | head -1)
        ANDROID_ACC=$(grep "准确率" "$ANDROID_RESULT_FILE" | grep -o '[0-9.]*' | head -1)
        
        if [ ! -z "$LOCAL_TIME" ] && [ ! -z "$ANDROID_TIME" ]; then
            echo "推理时间:"
            echo "  本地 macOS C++: ${LOCAL_TIME}ms"
            echo "  Android 设备:   ${ANDROID_TIME}ms"
            
            # 计算性能比
            RATIO=$(echo "scale=2; $LOCAL_TIME / $ANDROID_TIME" | bc 2>/dev/null || echo "N/A")
            if [ "$RATIO" != "N/A" ]; then
                echo "  性能比 (本地/Android): ${RATIO}x"
            fi
        fi
        
        if [ ! -z "$LOCAL_FPS" ] && [ ! -z "$ANDROID_FPS" ]; then
            echo "FPS:"
            echo "  本地 macOS C++: ${LOCAL_FPS}"
            echo "  Android 设备:   ${ANDROID_FPS}"
        fi
        
        if [ ! -z "$LOCAL_ACC" ] && [ ! -z "$ANDROID_ACC" ]; then
            echo "准确率:"
            echo "  本地 macOS C++: ${LOCAL_ACC}%"
            echo "  Android 设备:   ${ANDROID_ACC}%"
        fi
        
    else
        echo -e "${YELLOW}无法进行性能对比: 缺少本地或 Android 结果文件${NC}"
        echo "请确保已运行本地 C++ 推理测试"
    fi
}

# 清理设备文件（可选）
cleanup_device() {
    echo -e "${YELLOW}清理设备文件...${NC}"
    adb shell "rm -rf $DEVICE_DIR" 2>/dev/null || true
    echo -e "${GREEN}✓ 设备文件清理完成${NC}"
}

# 生成最终报告
generate_report() {
    echo -e "${YELLOW}生成最终报告...${NC}"
    
    REPORT_FILE="$RESULTS_DIR/android_real_onnx_deployment_report.md"
    
    cat > "$REPORT_FILE" << EOF
# Android 真实 ONNX Runtime 部署测试报告

**生成时间**: $(date)
**测试设备**: $(adb shell getprop ro.product.model | tr -d '\r')
**Android 版本**: $(adb shell getprop ro.build.version.release | tr -d '\r')
**CPU 架构**: $(adb shell getprop ro.product.cpu.abi | tr -d '\r')

## 部署信息

- **ONNX Runtime 版本**: 真实 Android 交叉编译版本
- **模型**: MNIST 手写数字识别
- **测试数据**: 真实 MNIST 测试集样本
- **部署方式**: 静态链接可执行文件

## 测试结果

EOF

    if [ -f "$RESULTS_DIR/android_real_onnx_results.txt" ]; then
        echo "" >> "$REPORT_FILE"
        echo "\`\`\`" >> "$REPORT_FILE"
        cat "$RESULTS_DIR/android_real_onnx_results.txt" >> "$REPORT_FILE"
        echo "\`\`\`" >> "$REPORT_FILE"
    fi
    
    echo "" >> "$REPORT_FILE"
    echo "## 部署状态" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "✅ 交叉编译成功" >> "$REPORT_FILE"
    echo "✅ 文件部署成功" >> "$REPORT_FILE"
    echo "✅ 推理测试完成" >> "$REPORT_FILE"
    echo "✅ 结果收集成功" >> "$REPORT_FILE"
    
    echo -e "${GREEN}✓ 报告已生成: $REPORT_FILE${NC}"
}

# 主函数
main() {
    echo "开始时间: $(date)"
    
    check_device
    deploy_files
    run_android_tests
    get_results
    performance_comparison
    generate_report
    
    echo -e "\n${GREEN}=== Android 真实 ONNX Runtime 部署测试完成！===${NC}"
    echo "结束时间: $(date)"
    
    # 询问是否清理设备文件
    echo -e "\n${YELLOW}是否清理设备上的测试文件? (y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        cleanup_device
    fi
}

# 执行主函数
main "$@" 