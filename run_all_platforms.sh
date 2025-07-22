#!/bin/bash

echo "🚀 === 跨平台 MNIST 推理完整测试流程 ==="
echo "将依次运行: 本地Python/C/C++ + Android C/C++ 共5个配置"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 步骤计数器
STEP=1

print_step() {
    echo -e "${BLUE}=== 步骤 $STEP: $1 ===${NC}"
    ((STEP++))
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查是否在正确的目录
if [ ! -f "train/train_model.py" ]; then
    print_error "请在DL2C项目根目录运行此脚本"
    exit 1
fi

print_step "环境检查"
echo "检查必要的文件和依赖..."

# 检查Python依赖
python -c "import torch, onnx, onnxruntime, numpy, matplotlib" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "Python依赖检查通过"
else
    print_error "Python依赖缺失，请运行: pip install torch onnx onnxruntime numpy matplotlib"
    exit 1
fi

# 检查Android设备
adb devices | grep -q "device$"
if [ $? -eq 0 ]; then
    print_success "Android设备连接正常"
    DEVICE_MODEL=$(adb shell getprop ro.product.model 2>/dev/null | tr -d '\r')
    echo "  设备型号: $DEVICE_MODEL"
else
    print_warning "Android设备未连接，将跳过Android测试"
    SKIP_ANDROID=true
fi

print_step "模型训练和导出"
echo "训练MNIST模型并导出为ONNX格式..."

cd train
if [ ! -f "../models/mnist_model.onnx" ]; then
    echo "正在训练模型..."
    python train_model.py
    if [ $? -ne 0 ]; then
        print_error "模型训练失败"
        exit 1
    fi
    
    echo "正在导出ONNX模型..."
    python export_onnx.py
    if [ $? -ne 0 ]; then
        print_error "ONNX导出失败"
        exit 1
    fi
    print_success "模型训练和导出完成"
else
    print_success "ONNX模型已存在，跳过训练"
fi
cd ..

print_step "生成测试数据"
echo "生成真实MNIST测试数据..."

if [ ! -d "test_data_mnist" ] || [ ! -f "test_data_mnist/metadata.json" ]; then
    python mnist_data_loader.py
    if [ $? -ne 0 ]; then
        print_error "测试数据生成失败"
        exit 1
    fi
    print_success "测试数据生成完成"
else
    print_success "测试数据已存在，跳过生成"
fi

print_step "本地推理测试 (1/3) - Python版本"
echo "运行Python MNIST推理..."

cd inference
python python_inference_mnist.py
if [ $? -eq 0 ]; then
    print_success "Python推理测试完成"
else
    print_error "Python推理测试失败"
fi

print_step "编译本地C/C++版本"
echo "编译本地推理程序..."

cd ../build
if [ ! -d "build_macos" ]; then
    mkdir build_macos
fi

cd build_macos
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4

if [ $? -eq 0 ]; then
    print_success "本地C/C++编译完成"
else
    print_error "本地C/C++编译失败"
    exit 1
fi

print_step "本地推理测试 (2/3) - C++版本"
echo "运行C++ MNIST推理..."

./bin/mnist_inference_cpp_mnist
if [ $? -eq 0 ]; then
    print_success "C++推理测试完成"
else
    print_error "C++推理测试失败"
fi

print_step "本地推理测试 (3/3) - C版本"
echo "运行C MNIST推理..."

./bin/mnist_inference_c_mnist
if [ $? -eq 0 ]; then
    print_success "C推理测试完成"
else
    print_error "C推理测试失败"
fi

cd ../..

if [ "$SKIP_ANDROID" != "true" ]; then
    print_step "Android跨平台编译"
    echo "编译Android版本推理程序..."

    cd build
    export ANDROID_NDK_HOME="/opt/homebrew/share/android-ndk"
    export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"
    
    ./build_android_real_onnx.sh
    if [ $? -eq 0 ]; then
        print_success "Android编译完成"
    else
        print_error "Android编译失败"
        cd ..
        SKIP_ANDROID=true
    fi

    if [ "$SKIP_ANDROID" != "true" ]; then
        print_step "Android推理测试 (4/5 & 5/5) - C++和C版本"
        echo "部署并运行Android推理测试..."

        ./deploy_and_test_real_onnx.sh
        if [ $? -eq 0 ]; then
            print_success "Android推理测试完成"
        else
            print_error "Android推理测试失败"
        fi
        cd ..
    fi
else
    print_warning "跳过Android测试"
fi

print_step "跨平台性能分析"
echo "生成全面的性能对比报告..."

python android_cross_platform_analysis.py
if [ $? -eq 0 ]; then
    print_success "性能分析完成"
else
    print_error "性能分析失败"
fi

print_step "结果展示"
echo ""
echo "🎉 === 跨平台测试完成！==="
echo ""

# 检查生成的结果文件
RESULTS_DIR="results"
LOCAL_CONFIGS=0
ANDROID_CONFIGS=0

echo "📊 测试结果汇总:"
if [ -f "$RESULTS_DIR/python_inference_mnist_results.json" ]; then
    echo "  ✓ Python版本结果"
    ((LOCAL_CONFIGS++))
fi

if [ -f "$RESULTS_DIR/c_inference_mnist_results.json" ]; then
    echo "  ✓ C语言版本结果"
    ((LOCAL_CONFIGS++))
fi

if [ -f "$RESULTS_DIR/cpp_inference_mnist_results.json" ]; then
    echo "  ✓ C++版本结果"
    ((LOCAL_CONFIGS++))
fi

if [ -f "$RESULTS_DIR/android_real_onnx_results.txt" ]; then
    echo "  ✓ Android C++版本结果"
    ((ANDROID_CONFIGS++))
fi

if [ -f "$RESULTS_DIR/android_real_onnx_c_results.txt" ]; then
    echo "  ✓ Android C版本结果"
    ((ANDROID_CONFIGS++))
fi

TOTAL_CONFIGS=$((LOCAL_CONFIGS + ANDROID_CONFIGS))
echo ""
echo "📈 成功完成 $TOTAL_CONFIGS/5 个配置的测试"
echo "   - 本地配置: $LOCAL_CONFIGS/3"
echo "   - Android配置: $ANDROID_CONFIGS/2"
echo ""

if [ -f "$RESULTS_DIR/comprehensive_cross_platform_report.md" ]; then
    echo "📋 详细报告: $RESULTS_DIR/comprehensive_cross_platform_report.md"
fi

if [ -f "$RESULTS_DIR/comprehensive_cross_platform_analysis.png" ]; then
    echo "📊 可视化图表: $RESULTS_DIR/comprehensive_cross_platform_analysis.png"
    echo ""
    echo "🖼️  查看图表 (macOS): open $RESULTS_DIR/comprehensive_cross_platform_analysis.png"
fi

echo ""
echo "💡 如需重新运行特定测试:"
echo "   - Python: python inference/python_inference_mnist.py"
echo "   - C/C++: cd build/build_macos && make && ./bin/mnist_inference_cpp_mnist"
echo "   - Android: cd build && ./deploy_and_test_real_onnx.sh"
echo "   - 分析: python android_cross_platform_analysis.py"
echo ""
print_success "跨平台MNIST推理测试流程完成！" 