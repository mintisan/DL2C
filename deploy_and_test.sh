#!/bin/bash
# 统一版本 ONNX Runtime 部署和测试脚本
# 支持 Android 部署测试和 macOS 本地测试

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== 统一版本 ONNX Runtime 部署和测试脚本 ===${NC}"

# 设置变量
PROJECT_DIR=$(pwd)/..
ANDROID_ABI="arm64-v8a"
ANDROID_EXE_DIR="android_executables/${ANDROID_ABI}"
DEVICE_DIR="/data/local/tmp/mnist_onnx"
RESULTS_DIR="results"

# 默认测试Android版本，除非指定macos
TEST_TARGET="android"
if [[ "$1" == "macos" ]]; then
    TEST_TARGET="macos"
fi

echo -e "${YELLOW}测试目标: $TEST_TARGET${NC}"

# 检查macOS可执行文件
check_macos_executables() {
    echo -e "${YELLOW}检查macOS可执行文件...${NC}"
    
    MACOS_EXE_DIR="inference"
    
    if [ ! -f "$MACOS_EXE_DIR/cpp_inference" ]; then
        echo -e "${RED}错误: macOS C++ 统一可执行文件不存在${NC}"
        echo "请先运行: ./build.sh macos"
        return 1
    fi
    
    if [ ! -f "$MACOS_EXE_DIR/c_inference" ]; then
        echo -e "${RED}错误: macOS C 统一可执行文件不存在${NC}"
        echo "请先运行: ./build.sh macos"
        return 1
    fi
    
    echo -e "${GREEN}✓ macOS 统一可执行文件检查通过${NC}"
    return 0
}

# 检查设备连接
check_device() {
    echo -e "${YELLOW}检查设备连接...${NC}"
    
    DEVICE_COUNT=$(adb devices 2>/dev/null | grep -c "device$" || echo "0")
    if [ "$DEVICE_COUNT" -eq 0 ] 2>/dev/null; then
        echo -e "${RED}错误: 没有连接的 Android 设备${NC}"
        return 1
    fi
    
    DEVICE_ID=$(adb devices | grep "device$" | head -1 | awk '{print $1}')
    echo -e "${GREEN}✓ 使用设备: $DEVICE_ID${NC}"
    
    # 获取设备信息
    echo -e "${BLUE}设备信息:${NC}"
    echo "  型号: $(adb shell getprop ro.product.model | tr -d '\r')"
    echo "  版本: Android $(adb shell getprop ro.build.version.release | tr -d '\r')"
    echo "  架构: $(adb shell getprop ro.product.cpu.abi | tr -d '\r')"
    echo "  内核: $(adb shell uname -r | tr -d '\r')"
    
    return 0
}

# 部署文件到设备
deploy_files() {
    echo -e "${YELLOW}部署文件到 Android 设备...${NC}"
    
    # 创建设备目录
    adb shell "mkdir -p $DEVICE_DIR" 2>/dev/null || true
    
    # 检查可执行文件
    if [ ! -f "$ANDROID_EXE_DIR/cpp_inference" ]; then
        echo -e "${RED}错误: Android C++ 统一可执行文件不存在${NC}"
        echo "请先运行: ./build.sh android"
        return 1
    fi
    
    if [ ! -f "$ANDROID_EXE_DIR/c_inference" ]; then
        echo -e "${RED}错误: Android C 统一可执行文件不存在${NC}"
        echo "请先运行: ./build.sh android"
        return 1
    fi
    
    # 推送C++可执行文件
    echo -e "${BLUE}推送 C++ 统一可执行文件...${NC}"
    adb push "$ANDROID_EXE_DIR/cpp_inference" "$DEVICE_DIR/"
    adb shell "chmod +x $DEVICE_DIR/cpp_inference"
    
    # 推送C可执行文件
    echo -e "${BLUE}推送 C 统一可执行文件...${NC}"
    adb push "$ANDROID_EXE_DIR/c_inference" "$DEVICE_DIR/"
    adb shell "chmod +x $DEVICE_DIR/c_inference"
    
    # 推送 ONNX 模型
    if [ -f "models/mnist_model.onnx" ]; then
        echo -e "${BLUE}推送 ONNX 模型...${NC}"
        adb shell "mkdir -p $DEVICE_DIR/models" 2>/dev/null || true
        adb push "models/mnist_model.onnx" "$DEVICE_DIR/models/"
    else
        echo -e "${YELLOW}警告: ONNX 模型文件不存在，程序将使用模拟数据${NC}"
    fi
    
    # 推送测试数据
    if [ -d "test_data" ]; then
        echo -e "${BLUE}推送测试数据...${NC}"
        adb shell "mkdir -p $DEVICE_DIR/test_data" 2>/dev/null || true
        adb push "test_data/." "$DEVICE_DIR/test_data/"
    else
        echo -e "${YELLOW}警告: MNIST 测试数据不存在，程序将使用模拟数据${NC}"
    fi
    
    # 创建结果目录
    adb shell "mkdir -p $DEVICE_DIR/results" 2>/dev/null || true
    
    echo -e "${GREEN}✓ 文件部署完成${NC}"
    return 0
}

# 运行 macOS 测试
run_macos_tests() {
    echo -e "${YELLOW}在 macOS 上运行统一版本测试...${NC}"
    
    MACOS_EXE_DIR="inference"
    
    # 测试 C++ 版本
    echo -e "${BLUE}执行 macOS C++ 统一推理测试...${NC}"
    cd "$MACOS_EXE_DIR"
    ./cpp_inference
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ macOS C++ 统一测试完成${NC}"
    else
        echo -e "${RED}✗ macOS C++ 统一测试失败${NC}"
        cd - > /dev/null
        return 1
    fi
    
    # 测试 C 版本
    echo -e "\n${BLUE}执行 macOS C 统一推理测试...${NC}"
    ./c_inference
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ macOS C 统一测试完成${NC}"
    else
        echo -e "${RED}✗ macOS C 统一测试失败${NC}"
        cd - > /dev/null
        return 1
    fi
    
    cd - > /dev/null
    return 0
}

# 运行 Android 测试
run_android_tests() {
    echo -e "${YELLOW}在 Android 设备上运行统一版本测试...${NC}"
    
    # 测试 C++ 版本
    echo -e "${BLUE}执行 Android C++ 统一推理测试...${NC}"
    adb shell "cd $DEVICE_DIR && ./cpp_inference"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Android C++ 统一测试完成${NC}"
    else
        echo -e "${RED}✗ Android C++ 统一测试失败${NC}"
        return 1
    fi
    
    # 测试 C 版本
    echo -e "\n${BLUE}执行 Android C 统一推理测试...${NC}"
    adb shell "cd $DEVICE_DIR && ./c_inference"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Android C 统一测试完成${NC}"
    else
        echo -e "${RED}✗ Android C 统一测试失败${NC}"
        return 1
    fi
    
    return 0
}

# 获取测试结果
get_results() {
    echo -e "${YELLOW}获取测试结果...${NC}"
    
    # 确保本地结果目录存在
    mkdir -p "$RESULTS_DIR"
    
    if [[ "$TEST_TARGET" == "android" ]]; then
        # 拉取Android C++版本结果文件
        if adb shell "test -f $DEVICE_DIR/results/android_unified_cpp_results.txt"; then
            adb pull "$DEVICE_DIR/results/android_unified_cpp_results.txt" "$RESULTS_DIR/"
            echo -e "${GREEN}✓ Android C++ 统一结果文件已下载${NC}"
        else
            echo -e "${YELLOW}警告: Android C++ 统一结果文件不存在${NC}"
        fi
        
        # 拉取Android C版本结果文件
        if adb shell "test -f $DEVICE_DIR/results/android_unified_c_results.txt"; then
            adb pull "$DEVICE_DIR/results/android_unified_c_results.txt" "$RESULTS_DIR/"
            echo -e "${GREEN}✓ Android C 统一结果文件已下载${NC}"
        else
            echo -e "${YELLOW}警告: Android C 统一结果文件不存在${NC}"
        fi
        
        # 显示Android结果
        if [ -f "$RESULTS_DIR/android_unified_cpp_results.txt" ]; then
            echo -e "${BLUE}Android C++ 统一测试结果:${NC}"
            cat "$RESULTS_DIR/android_unified_cpp_results.txt"
        fi
        
        if [ -f "$RESULTS_DIR/android_unified_c_results.txt" ]; then
            echo -e "\n${BLUE}Android C 统一测试结果:${NC}"
            cat "$RESULTS_DIR/android_unified_c_results.txt"
        fi
        
    else
        # macOS结果处理
        echo -e "${BLUE}macOS 统一测试结果:${NC}"
        
        # 显示macOS C++结果
        if [ -f "$RESULTS_DIR/macos_unified_cpp_results.txt" ]; then
            echo -e "${BLUE}macOS C++ 统一测试结果:${NC}"
            cat "$RESULTS_DIR/macos_unified_cpp_results.txt"
        fi
        
        # 显示macOS C结果
        if [ -f "$RESULTS_DIR/macos_unified_c_results.txt" ]; then
            echo -e "\n${BLUE}macOS C 统一测试结果:${NC}"
            cat "$RESULTS_DIR/macos_unified_c_results.txt"
        fi
    fi
}

# 性能对比
performance_comparison() {
    echo -e "\n${YELLOW}统一版本性能对比分析...${NC}"
    
    if [[ "$TEST_TARGET" == "android" ]]; then
        # Android vs 本地对比
        LOCAL_CPP_RESULT="results/macos_unified_cpp_results.txt"
        ANDROID_CPP_RESULT="$RESULTS_DIR/android_unified_cpp_results.txt"
        
        if [ -f "$LOCAL_CPP_RESULT" ] && [ -f "$ANDROID_CPP_RESULT" ]; then
            echo -e "${BLUE}=== macOS vs Android 统一版本性能对比 ===${NC}"
            
            # 提取性能数据
            LOCAL_TIME=$(grep "平均推理时间" "$LOCAL_CPP_RESULT" | grep -o '[0-9.]*' | head -1)
            ANDROID_TIME=$(grep "平均推理时间" "$ANDROID_CPP_RESULT" | grep -o '[0-9.]*' | head -1)
            
            LOCAL_FPS=$(grep "推理速度" "$LOCAL_CPP_RESULT" | grep -o '[0-9.]*' | head -1)
            ANDROID_FPS=$(grep "推理速度" "$ANDROID_CPP_RESULT" | grep -o '[0-9.]*' | head -1)
            
            LOCAL_ACC=$(grep "准确率" "$LOCAL_CPP_RESULT" | grep -o '[0-9.]*' | head -1)
            ANDROID_ACC=$(grep "准确率" "$ANDROID_CPP_RESULT" | grep -o '[0-9.]*' | head -1)
            
            if [ ! -z "$LOCAL_TIME" ] && [ ! -z "$ANDROID_TIME" ]; then
                echo "C++ 统一版本推理时间:"
                echo "  macOS:   ${LOCAL_TIME}ms"
                echo "  Android: ${ANDROID_TIME}ms"
                
                # 计算性能比
                if command -v bc >/dev/null 2>&1; then
                    RATIO=$(echo "scale=2; $LOCAL_TIME / $ANDROID_TIME" | bc 2>/dev/null || echo "N/A")
                    if [ "$RATIO" != "N/A" ]; then
                        echo "  性能比 (macOS/Android): ${RATIO}x"
                    fi
                fi
            fi
            
            if [ ! -z "$LOCAL_FPS" ] && [ ! -z "$ANDROID_FPS" ]; then
                echo "C++ 统一版本FPS:"
                echo "  macOS:   ${LOCAL_FPS}"
                echo "  Android: ${ANDROID_FPS}"
            fi
            
            if [ ! -z "$LOCAL_ACC" ] && [ ! -z "$ANDROID_ACC" ]; then
                echo "C++ 统一版本准确率:"
                echo "  macOS:   ${LOCAL_ACC}%"
                echo "  Android: ${ANDROID_ACC}%"
            fi
            
        else
            echo -e "${YELLOW}无法进行性能对比: 缺少本地或 Android 结果文件${NC}"
            echo "请确保已运行本地和Android测试"
        fi
        
    else
        # macOS C vs C++ 对比
        C_RESULT="$RESULTS_DIR/macos_unified_c_results.txt"
        CPP_RESULT="$RESULTS_DIR/macos_unified_cpp_results.txt"
        
        if [ -f "$C_RESULT" ] && [ -f "$CPP_RESULT" ]; then
            echo -e "${BLUE}=== macOS C vs C++ 统一版本性能对比 ===${NC}"
            
            C_TIME=$(grep "平均推理时间" "$C_RESULT" | grep -o '[0-9.]*' | head -1)
            CPP_TIME=$(grep "平均推理时间" "$CPP_RESULT" | grep -o '[0-9.]*' | head -1)
            
            if [ ! -z "$C_TIME" ] && [ ! -z "$CPP_TIME" ]; then
                echo "macOS 统一版本推理时间:"
                echo "  C 版本:   ${C_TIME}ms"
                echo "  C++ 版本: ${CPP_TIME}ms"
            fi
        fi
    fi
}

# 清理设备文件（可选）
cleanup_device() {
    if [[ "$TEST_TARGET" == "android" ]]; then
        echo -e "${YELLOW}清理Android设备文件...${NC}"
        adb shell "rm -rf $DEVICE_DIR" 2>/dev/null || true
        echo -e "${GREEN}✓ Android设备文件清理完成${NC}"
    fi
}

# 生成最终报告
generate_report() {
    echo -e "${YELLOW}生成最终报告...${NC}"
    
    REPORT_FILE="$RESULTS_DIR/unified_deployment_report.md"
    
    cat > "$REPORT_FILE" << EOF
# 统一版本 ONNX Runtime 部署测试报告

**生成时间**: $(date)
**测试目标**: $TEST_TARGET

EOF

    if [[ "$TEST_TARGET" == "android" ]]; then
        cat >> "$REPORT_FILE" << EOF
**测试设备**: $(adb shell getprop ro.product.model | tr -d '\r')
**Android 版本**: $(adb shell getprop ro.build.version.release | tr -d '\r')
**CPU 架构**: $(adb shell getprop ro.product.cpu.abi | tr -d '\r')

## 部署信息

- **ONNX Runtime 版本**: Android 交叉编译版本
- **模型**: MNIST 手写数字识别
- **统一代码**: 同一份源码支持macOS和Android
- **部署方式**: 静态链接可执行文件

## Android 测试结果

EOF
        if [ -f "$RESULTS_DIR/android_unified_cpp_results.txt" ]; then
            echo "### C++ 统一版本" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
            echo "\`\`\`" >> "$REPORT_FILE"
            cat "$RESULTS_DIR/android_unified_cpp_results.txt" >> "$REPORT_FILE"
            echo "\`\`\`" >> "$REPORT_FILE"
        fi
        
        if [ -f "$RESULTS_DIR/android_unified_c_results.txt" ]; then
            echo "### C 统一版本" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
            echo "\`\`\`" >> "$REPORT_FILE"
            cat "$RESULTS_DIR/android_unified_c_results.txt" >> "$REPORT_FILE"
            echo "\`\`\`" >> "$REPORT_FILE"
        fi
        
    else
        cat >> "$REPORT_FILE" << EOF
**测试平台**: macOS $(sw_vers -productVersion)
**处理器**: $(sysctl -n machdep.cpu.brand_string)

## macOS 测试结果

EOF
        if [ -f "$RESULTS_DIR/macos_unified_cpp_results.txt" ]; then
            echo "### C++ 统一版本" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
            echo "\`\`\`" >> "$REPORT_FILE"
            cat "$RESULTS_DIR/macos_unified_cpp_results.txt" >> "$REPORT_FILE"
            echo "\`\`\`" >> "$REPORT_FILE"
        fi
        
        if [ -f "$RESULTS_DIR/macos_unified_c_results.txt" ]; then
            echo "### C 统一版本" >> "$REPORT_FILE"
            echo "" >> "$REPORT_FILE"
            echo "\`\`\`" >> "$REPORT_FILE"
            cat "$RESULTS_DIR/macos_unified_c_results.txt" >> "$REPORT_FILE"
            echo "\`\`\`" >> "$REPORT_FILE"
        fi
    fi
    
    echo "" >> "$REPORT_FILE"
    echo "## 部署状态" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "✅ 统一代码编译成功" >> "$REPORT_FILE"
    echo "✅ 跨平台部署成功" >> "$REPORT_FILE"
    echo "✅ 推理测试完成" >> "$REPORT_FILE"
    echo "✅ 结果收集成功" >> "$REPORT_FILE"
    
    echo -e "${GREEN}✓ 报告已生成: $REPORT_FILE${NC}"
}

# 显示使用说明
show_usage() {
    echo "使用方法:"
    echo "  $0 [macos|android]"
    echo ""
    echo "参数:"
    echo "  macos    - 在macOS上测试统一版本"
    echo "  android  - 部署到Android设备并测试 (默认)"
    echo ""
    echo "示例:"
    echo "  $0 macos     # macOS本地测试"
    echo "  $0 android   # Android部署测试"
    echo "  $0           # 默认Android部署测试"
}

# 主函数
main() {
    echo "开始时间: $(date)"
    echo "测试目标: $TEST_TARGET"
    
    if [[ "$TEST_TARGET" == "android" ]]; then
        if ! check_device; then
            echo -e "${RED}Android设备检查失败${NC}"
            exit 1
        fi
        
        if ! deploy_files; then
            echo -e "${RED}Android文件部署失败${NC}"
            exit 1
        fi
        
        if ! run_android_tests; then
            echo -e "${RED}Android测试失败${NC}"
            exit 1
        fi
        
    else
        if ! check_macos_executables; then
            echo -e "${RED}macOS可执行文件检查失败${NC}"
            exit 1
        fi
        
        if ! run_macos_tests; then
            echo -e "${RED}macOS测试失败${NC}"
            exit 1
        fi
    fi
    
    get_results
    performance_comparison
    generate_report
    
    echo -e "\n${GREEN}=== $TEST_TARGET 统一版本测试完成！===${NC}"
    echo "结束时间: $(date)"
    
    # 询问是否清理设备文件
    if [[ "$TEST_TARGET" == "android" ]]; then
        echo -e "\n${YELLOW}是否清理Android设备上的测试文件? (Y/n)${NC}"
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cleanup_device
        fi
    fi
}

# 处理命令行参数
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

if [[ -n "$1" ]] && [[ "$1" != "macos" ]] && [[ "$1" != "android" ]]; then
    echo -e "${RED}错误: 无效的测试目标 '$1'${NC}"
    show_usage
    exit 1
fi

# 执行主函数
main "$@" 