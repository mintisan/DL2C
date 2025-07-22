#!/bin/bash

echo "🔍 === 跨平台环境快速检查 ==="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 检查环境
echo "📋 环境检查:"

# Python依赖
python -c "import torch, onnx, onnxruntime, numpy, matplotlib" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "Python依赖完整"
    PYTHON_VERSION=$(python -c "import torch; print(f'PyTorch {torch.__version__}')")
    echo "    $PYTHON_VERSION"
else
    print_error "Python依赖缺失"
fi

# Android设备
adb devices | grep -q "device$"
if [ $? -eq 0 ]; then
    DEVICE_MODEL=$(adb shell getprop ro.product.model 2>/dev/null | tr -d '\r')
    print_success "Android设备已连接: $DEVICE_MODEL"
else
    print_warning "Android设备未连接"
fi

# ONNX模型
if [ -f "models/mnist_model.onnx" ]; then
    SIZE=$(ls -lh models/mnist_model.onnx | awk '{print $5}')
    print_success "ONNX模型存在 ($SIZE)"
else
    print_warning "ONNX模型不存在，需要训练"
fi

# 测试数据
if [ -f "test_data_mnist/metadata.json" ]; then
    SAMPLES=$(grep "num_samples" test_data_mnist/metadata.json | grep -o '[0-9]*')
    print_success "测试数据存在 ($SAMPLES 样本)"
else
    print_warning "测试数据不存在，需要生成"
fi

echo ""
echo "📊 现有结果检查:"

# 检查结果文件
RESULTS_DIR="results"
FOUND_RESULTS=0

if [ -f "$RESULTS_DIR/python_inference_mnist_results.json" ]; then
    print_success "Python结果文件存在"
    ((FOUND_RESULTS++))
else
    print_warning "Python结果文件缺失"
fi

if [ -f "$RESULTS_DIR/c_inference_mnist_results.json" ]; then
    print_success "C语言结果文件存在"
    ((FOUND_RESULTS++))
else
    print_warning "C语言结果文件缺失"
fi

if [ -f "$RESULTS_DIR/cpp_inference_mnist_results.json" ]; then
    print_success "C++结果文件存在"
    ((FOUND_RESULTS++))
else
    print_warning "C++结果文件缺失"
fi

if [ -f "$RESULTS_DIR/android_real_onnx_results.txt" ]; then
    print_success "Android C++结果文件存在"
    ((FOUND_RESULTS++))
else
    print_warning "Android C++结果文件缺失"
fi

if [ -f "$RESULTS_DIR/android_real_onnx_c_results.txt" ]; then
    print_success "Android C结果文件存在"
    ((FOUND_RESULTS++))
else
    print_warning "Android C结果文件缺失"
fi

echo ""
echo "📈 结果汇总: $FOUND_RESULTS/5 个配置已完成"

if [ $FOUND_RESULTS -eq 5 ]; then
    print_success "🎉 所有配置都已完成！"
    echo ""
    print_info "可以直接运行性能分析："
    echo "    python android_cross_platform_analysis.py"
    echo ""
    print_info "查看现有结果："
    echo "    open results/comprehensive_cross_platform_analysis.png"
elif [ $FOUND_RESULTS -gt 0 ]; then
    print_info "部分测试已完成，可运行分析："
    echo "    python android_cross_platform_analysis.py"
    echo ""
    print_info "运行完整测试："
    echo "    ./run_all_platforms.sh"
else
    print_info "开始完整测试流程："
    echo "    ./run_all_platforms.sh"
fi

echo ""
echo "🚀 快速执行选项："
echo "  📊 生成当前分析: python android_cross_platform_analysis.py"
echo "  🔄 完整测试流程: ./run_all_platforms.sh" 
echo "  📋 查看详细指南: cat EXECUTION_GUIDE.md" 