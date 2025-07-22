#!/bin/bash
# Android 部署和测试脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Android MNIST 推理部署和测试脚本 ===${NC}"

# 设置变量
PROJECT_DIR=$(pwd)/..
ANDROID_ABI="arm64-v8a"
ANDROID_EXE_DIR="../android_executables/${ANDROID_ABI}"
DEVICE_DIR="/data/local/tmp/mnist_inference"
RESULTS_DIR="../results"

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
    echo "  型号: $(adb shell getprop ro.product.model)"
    echo "  Android版本: $(adb shell getprop ro.build.version.release)"
    echo "  CPU架构: $(adb shell getprop ro.product.cpu.abi)"
}

# 准备设备目录
prepare_device() {
    echo -e "${YELLOW}准备设备目录...${NC}"
    
    # 创建工作目录
    adb shell "mkdir -p $DEVICE_DIR"
    adb shell "mkdir -p $DEVICE_DIR/models"
    adb shell "mkdir -p $DEVICE_DIR/test_data"
    adb shell "mkdir -p $DEVICE_DIR/results"
    
    echo -e "${GREEN}✓ 设备目录创建完成${NC}"
}

# 部署可执行文件
deploy_executables() {
    echo -e "${YELLOW}部署可执行文件到设备...${NC}"
    
    if [ ! -d "$ANDROID_EXE_DIR" ]; then
        echo -e "${RED}错误: Android 可执行文件目录不存在: $ANDROID_EXE_DIR${NC}"
        echo "请先运行 ./build_android.sh 进行交叉编译"
        exit 1
    fi
    
    # 推送可执行文件
    for exe in "$ANDROID_EXE_DIR"/*; do
        if [ -f "$exe" ] && [ -x "$exe" ]; then
            filename=$(basename "$exe")
            echo "推送: $filename"
            adb push "$exe" "$DEVICE_DIR/"
            adb shell "chmod +x $DEVICE_DIR/$filename"
        fi
    done
    
    echo -e "${GREEN}✓ 可执行文件部署完成${NC}"
}

# 部署模型和数据文件
deploy_data() {
    echo -e "${YELLOW}部署模型和数据文件...${NC}"
    
    # 推送模型文件
    if [ -f "../models/mnist_model.onnx" ]; then
        echo "推送模型文件..."
        adb push "../models/mnist_model.onnx" "$DEVICE_DIR/models/"
    else
        echo -e "${RED}警告: 模型文件不存在${NC}"
    fi
    
    # 推送测试数据（选择前10个样本以节省空间）
    if [ -d "../test_data_mnist" ]; then
        echo "推送测试数据..."
        adb push "../test_data_mnist" "$DEVICE_DIR/"
    else
        echo -e "${RED}警告: 测试数据不存在${NC}"
    fi
    
    echo -e "${GREEN}✓ 数据文件部署完成${NC}"
}

# 运行推理测试
run_inference_test() {
    local executable=$1
    local test_name=$2
    
    echo -e "${BLUE}=== 运行 $test_name 推理测试 ===${NC}"
    
    if adb shell "test -f $DEVICE_DIR/$executable"; then
        echo "运行 $executable..."
        
        # 记录开始时间
        start_time=$(date +%s)
        
        # 运行推理并捕获输出
        adb shell "cd $DEVICE_DIR && ./$executable" > "$RESULTS_DIR/android_${test_name}_output.txt" 2>&1
        
        # 记录结束时间
        end_time=$(date +%s)
        execution_time=$((end_time - start_time))
        
        echo "执行时间: ${execution_time}秒"
        
        # 显示部分输出
        echo -e "${YELLOW}输出摘要:${NC}"
        head -20 "$RESULTS_DIR/android_${test_name}_output.txt"
        
        # 检查是否有错误
        if grep -q "错误\|Error\|error" "$RESULTS_DIR/android_${test_name}_output.txt"; then
            echo -e "${RED}⚠ 检测到错误信息${NC}"
        else
            echo -e "${GREEN}✓ $test_name 推理完成${NC}"
        fi
        
    else
        echo -e "${RED}错误: 可执行文件不存在: $executable${NC}"
    fi
}

# 从设备拉取结果
pull_results() {
    echo -e "${YELLOW}拉取设备上的结果文件...${NC}"
    
    # 拉取结果文件（如果存在）
    adb shell "ls $DEVICE_DIR/results/" 2>/dev/null | while read -r file; do
        if [ -n "$file" ]; then
            echo "拉取结果文件: $file"
            adb pull "$DEVICE_DIR/results/$file" "$RESULTS_DIR/android_$file"
        fi
    done
    
    echo -e "${GREEN}✓ 结果文件拉取完成${NC}"
}

# 性能分析
analyze_performance() {
    echo -e "${BLUE}=== Android 性能分析 ===${NC}"
    
    # 获取设备CPU信息
    echo -e "${YELLOW}设备CPU信息:${NC}"
    adb shell "cat /proc/cpuinfo | grep -E 'processor|model name|cpu MHz|cache size' | head -20"
    
    # 获取内存信息
    echo -e "${YELLOW}内存信息:${NC}"
    adb shell "cat /proc/meminfo | grep -E 'MemTotal|MemFree|MemAvailable'"
    
    # 分析输出文件中的性能数据
    echo -e "${YELLOW}推理性能摘要:${NC}"
    
    for output_file in "$RESULTS_DIR"/android_*_output.txt; do
        if [ -f "$output_file" ]; then
            filename=$(basename "$output_file")
            echo ""
            echo "=== $filename ==="
            
            # 提取关键性能指标
            grep -E "平均推理时间|Average inference time|准确率|Accuracy|FPS|fps" "$output_file" || echo "未找到性能指标"
        fi
    done
}

# 与本地结果对比
compare_with_local() {
    echo -e "${BLUE}=== 与本地结果对比 ===${NC}"
    
    # 检查是否有本地结果文件
    local local_cpp_results="../results/cpp_inference_results.json"
    local local_c_results="../results/c_inference_results.json"
    
    if [ -f "$local_cpp_results" ] || [ -f "$local_c_results" ]; then
        echo -e "${YELLOW}本地推理结果:${NC}"
        
        if [ -f "$local_cpp_results" ]; then
            echo "C++ 本地结果:"
            cat "$local_cpp_results" | grep -E "accuracy|average_inference_time" || echo "未找到指标"
        fi
        
        if [ -f "$local_c_results" ]; then
            echo "C 本地结果:"
            cat "$local_c_results" | grep -E "accuracy|average_inference_time" || echo "未找到指标"
        fi
        
        echo ""
        echo -e "${YELLOW}对比说明:${NC}"
        echo "1. Android 设备通常比桌面CPU性能较低"
        echo "2. 交叉编译的优化可能不同"
        echo "3. 不同的浮点处理可能导致精度差异"
        
    else
        echo "未找到本地结果文件进行对比"
    fi
}

# 清理设备
cleanup_device() {
    echo -e "${YELLOW}清理设备临时文件...${NC}"
    
    read -p "是否要清理设备上的临时文件？(y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        adb shell "rm -rf $DEVICE_DIR"
        echo -e "${GREEN}✓ 设备清理完成${NC}"
    else
        echo "保留设备文件在: $DEVICE_DIR"
    fi
}

# 生成测试报告
generate_report() {
    echo -e "${YELLOW}生成测试报告...${NC}"
    
    local report_file="$RESULTS_DIR/android_test_report.md"
    
    cat > "$report_file" << EOF
# Android MNIST 推理测试报告

生成时间: $(date)

## 设备信息
- 设备ID: $(adb devices | grep "device$" | head -1 | awk '{print $1}')
- 型号: $(adb shell getprop ro.product.model)
- Android版本: $(adb shell getprop ro.build.version.release)
- CPU架构: $(adb shell getprop ro.product.cpu.abi)

## 测试结果

### C++ 推理结果
EOF

    if [ -f "$RESULTS_DIR/android_cpp_output.txt" ]; then
        echo "\`\`\`" >> "$report_file"
        tail -20 "$RESULTS_DIR/android_cpp_output.txt" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
    else
        echo "C++ 推理输出文件不存在" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

### C 推理结果
EOF

    if [ -f "$RESULTS_DIR/android_c_output.txt" ]; then
        echo "\`\`\`" >> "$report_file"
        tail -20 "$RESULTS_DIR/android_c_output.txt" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
    else
        echo "C 推理输出文件不存在" >> "$report_file"
    fi

    echo "" >> "$report_file"
    echo "## 性能对比" >> "$report_file"
    echo "详细的性能对比数据请查看具体的输出文件。" >> "$report_file"

    echo -e "${GREEN}✓ 测试报告生成: $report_file${NC}"
}

# 主函数
main() {
    echo "开始 Android 部署和测试流程..."
    
    # 确保结果目录存在
    mkdir -p "$RESULTS_DIR"
    
    check_device
    prepare_device
    deploy_executables
    deploy_data
    
    # 运行推理测试
    run_inference_test "mnist_inference_cpp" "cpp"
    run_inference_test "mnist_inference_c" "c"
    
    pull_results
    analyze_performance
    compare_with_local
    generate_report
    
    echo -e "${GREEN}=== Android 测试完成 ===${NC}"
    echo -e "${YELLOW}查看详细结果: $RESULTS_DIR/android_test_report.md${NC}"
    
    cleanup_device
}

# 运行主函数
main "$@" 