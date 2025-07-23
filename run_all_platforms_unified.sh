#!/bin/bash

echo "🚀 === 统一版本跨平台 MNIST 推理完整测试流程 ==="
echo "将依次运行: Python + 统一版本macOS C/C++ + 统一版本Android C/C++ 共5个配置"
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
    SKIP_ANDROID=false
else
    print_warning "Android设备未连接，将跳过Android测试"
    SKIP_ANDROID=true
fi

# 检查编译环境
print_step "编译环境检查"
echo "检查macOS和Android编译环境..."

# 检查macOS编译环境
if ! command -v cmake &> /dev/null; then
    print_error "cmake 未安装，macOS编译将跳过"
    SKIP_MACOS_BUILD=true
else
    print_success "cmake 已安装"
    SKIP_MACOS_BUILD=false
fi

# 检查Android NDK
ANDROID_NDK_HOME=${ANDROID_NDK_HOME:-"/opt/homebrew/share/android-ndk"}
if [ ! -d "$ANDROID_NDK_HOME" ] && [ "$SKIP_ANDROID" = false ]; then
    print_warning "Android NDK 未找到，Android编译将跳过"
    SKIP_ANDROID=true
fi

if [ "$SKIP_ANDROID" = false ]; then
    print_success "Android NDK 可用: $ANDROID_NDK_HOME"
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

print_step "本地推理测试 (1/5) - Python版本"
echo "运行Python MNIST推理..."

cd inference
python python_inference_mnist.py
if [ $? -eq 0 ]; then
    print_success "Python推理测试完成"
else
    print_error "Python推理测试失败"
fi
cd ..

print_step "编译统一版本macOS"
echo "编译统一版本macOS推理程序..."

if [ "$SKIP_MACOS_BUILD" = false ]; then
    cd build
    ./build_unified.sh macos
    if [ $? -eq 0 ]; then
        print_success "统一版本macOS编译完成"
    else
        print_error "统一版本macOS编译失败"
        SKIP_MACOS_BUILD=true
    fi
    cd ..
else
    print_warning "跳过macOS编译"
fi

print_step "本地推理测试 (2/5 & 3/5) - 统一版本macOS C++和C"
echo "运行统一版本macOS推理测试..."

if [ "$SKIP_MACOS_BUILD" = false ]; then
    cd build
    ./deploy_and_test_unified.sh macos
    if [ $? -eq 0 ]; then
        print_success "统一版本macOS推理测试完成"
    else
        print_error "统一版本macOS推理测试失败"
    fi
    cd ..
else
    print_warning "跳过统一版本macOS测试"
fi

print_step "编译统一版本Android"
echo "编译统一版本Android推理程序..."

if [ "$SKIP_ANDROID" = false ]; then
    cd build
    export ANDROID_NDK_HOME="$ANDROID_NDK_HOME"
    export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"
    
    ./build_unified.sh android
    if [ $? -eq 0 ]; then
        print_success "统一版本Android编译完成"
    else
        print_error "统一版本Android编译失败"
        SKIP_ANDROID=true
    fi
    cd ..
else
    print_warning "跳过统一版本Android编译"
fi

print_step "Android推理测试 (4/5 & 5/5) - 统一版本Android C++和C"
echo "部署并运行统一版本Android推理测试..."

if [ "$SKIP_ANDROID" = false ]; then
    cd build
    ./deploy_and_test_unified.sh android
    if [ $? -eq 0 ]; then
        print_success "统一版本Android推理测试完成"
    else
        print_error "统一版本Android推理测试失败"
    fi
    cd ..
else
    print_warning "跳过统一版本Android测试"
fi

print_step "统一版本跨平台性能分析"
echo "生成全面的统一版本性能对比报告..."

# 创建统一版本性能分析脚本
cat > unified_performance_analysis.py << 'EOF'
#!/usr/bin/env python3
"""
统一版本跨平台性能分析脚本
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_result_file(file_path):
    """加载结果文件"""
    if not os.path.exists(file_path):
        return None
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # 处理文本文件
        result = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 解析关键信息
        import re
        
        # 提取准确率
        acc_match = re.search(r'准确率:\s*([0-9.]+)%', content)
        if acc_match:
            result['accuracy'] = float(acc_match.group(1)) / 100
        
        # 提取平均推理时间
        time_match = re.search(r'平均推理时间:\s*([0-9.]+)\s*ms', content)
        if time_match:
            result['average_inference_time_ms'] = float(time_match.group(1))
        
        # 提取FPS
        fps_match = re.search(r'推理速度:\s*([0-9.]+)\s*FPS', content)
        if fps_match:
            result['fps'] = float(fps_match.group(1))
        
        # 提取样本数
        samples_match = re.search(r'总样本数:\s*([0-9]+)', content)
        if samples_match:
            result['total_samples'] = int(samples_match.group(1))
        
        return result

def generate_unified_analysis():
    """生成统一版本分析报告"""
    
    # 结果文件路径
    result_files = {
        'Python': 'results/python_inference_mnist_results.json',
        'macOS C++': 'results/macos_unified_cpp_results.txt',
        'macOS C': 'results/macos_unified_c_results.txt',
        'Android C++': 'results/android_unified_cpp_results.txt',
        'Android C': 'results/android_unified_c_results.txt'
    }
    
    # 加载结果
    results = {}
    for name, file_path in result_files.items():
        result = load_result_file(file_path)
        if result:
            results[name] = result
            print(f"✓ 加载 {name} 结果")
        else:
            print(f"⚠️  未找到 {name} 结果文件: {file_path}")
    
    if not results:
        print("❌ 没有找到任何结果文件")
        return
    
    # 生成可视化图表
    generate_unified_plots(results)
    
    # 生成文字报告
    generate_unified_report(results)

def generate_unified_plots(results):
    """生成统一版本可视化图表"""
    
    plt.style.use('default')
    
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Unified Cross-Platform MNIST Inference Performance Analysis', fontsize=16, fontweight='bold')
    
    # 颜色配置
    colors = {
        'Python': '#FF6B6B',
        'macOS C++': '#4ECDC4',
        'macOS C': '#45B7D1',
        'Android C++': '#96CEB4',
        'Android C': '#FECA57'
    }
    
    # 提取数据
    platforms = []
    accuracies = []
    times = []
    fps_values = []
    
    for name, result in results.items():
        platforms.append(name)
        
        # 处理不同格式的数据
        if 'summary' in result:
            # Python JSON格式
            accuracies.append(result['summary']['accuracy'] * 100)
            times.append(result['summary']['average_inference_time_ms'])
            fps_values.append(result['summary']['fps'])
        else:
            # 文本格式
            accuracies.append(result.get('accuracy', 0) * 100)
            times.append(result.get('average_inference_time_ms', 0))
            fps_values.append(result.get('fps', 0))
    
    # 1. 准确率对比
    bars1 = ax1.bar(platforms, accuracies, color=[colors.get(p, '#666666') for p in platforms])
    ax1.set_title('推理准确率对比', fontweight='bold')
    ax1.set_ylabel('准确率 (%)')
    ax1.set_ylim(0, 100)
    
    # 添加数值标签
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # 2. 推理时间对比
    bars2 = ax2.bar(platforms, times, color=[colors.get(p, '#666666') for p in platforms])
    ax2.set_title('平均推理时间对比', fontweight='bold')
    ax2.set_ylabel('推理时间 (ms)')
    
    # 添加数值标签
    for bar, time in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time:.2f}', ha='center', va='bottom')
    
    # 3. FPS对比
    bars3 = ax3.bar(platforms, fps_values, color=[colors.get(p, '#666666') for p in platforms])
    ax3.set_title('推理速度对比', fontweight='bold')
    ax3.set_ylabel('FPS')
    
    # 添加数值标签
    for bar, fps in zip(bars3, fps_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{fps:.1f}', ha='center', va='bottom')
    
    # 4. 平台分组对比
    macos_data = []
    android_data = []
    labels = []
    
    for name, result in results.items():
        if 'macOS' in name:
            if 'summary' in result:
                macos_data.append(result['summary']['average_inference_time_ms'])
            else:
                macos_data.append(result.get('average_inference_time_ms', 0))
            labels.append(name.replace('macOS ', ''))
        elif 'Android' in name:
            if 'summary' in result:
                android_data.append(result['summary']['average_inference_time_ms'])
            else:
                android_data.append(result.get('average_inference_time_ms', 0))
    
    if macos_data and android_data:
        x = np.arange(len(labels))
        width = 0.35
        
        ax4.bar(x - width/2, macos_data, width, label='macOS', color='#4ECDC4')
        ax4.bar(x + width/2, android_data, width, label='Android', color='#96CEB4')
        
        ax4.set_title('macOS vs Android 统一版本对比', fontweight='bold')
        ax4.set_ylabel('推理时间 (ms)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 旋转x轴标签以避免重叠
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', rotation=45)
    
    # 保存图表
    plt.savefig('results/unified_cross_platform_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 统一版本可视化图表已生成: results/unified_cross_platform_analysis.png")

def generate_unified_report(results):
    """生成统一版本文字报告"""
    
    report_file = 'results/unified_cross_platform_report.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 统一版本跨平台MNIST推理性能分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 测试概述\n\n")
        f.write("本报告展示了统一版本代码在不同平台上的MNIST推理性能对比。")
        f.write("统一版本使用相同的源代码，通过预处理器宏适配不同平台。\n\n")
        
        f.write("## 平台配置\n\n")
        f.write("| 平台 | 语言 | 编译方式 | 部署方式 |\n")
        f.write("|------|------|----------|----------|\n")
        
        for name in results.keys():
            if 'Python' in name:
                f.write(f"| {name} | Python | 解释执行 | 本地运行 |\n")
            elif 'macOS' in name:
                lang = 'C++' if 'C++' in name else 'C'
                f.write(f"| {name} | {lang} | 本地编译 | 本地运行 |\n")
            elif 'Android' in name:
                lang = 'C++' if 'C++' in name else 'C'
                f.write(f"| {name} | {lang} | 交叉编译 | 设备部署 |\n")
        
        f.write("\n## 性能结果\n\n")
        f.write("| 平台 | 准确率 | 平均时间(ms) | FPS | 样本数 |\n")
        f.write("|------|--------|--------------|-----|--------|\n")
        
        for name, result in results.items():
            if 'summary' in result:
                # Python JSON格式
                summary = result['summary']
                accuracy = summary['accuracy'] * 100
                time_ms = summary['average_inference_time_ms']
                fps = summary['fps']
                samples = summary['total_samples']
            else:
                # 文本格式
                accuracy = result.get('accuracy', 0) * 100
                time_ms = result.get('average_inference_time_ms', 0)
                fps = result.get('fps', 0)
                samples = result.get('total_samples', 0)
            
            f.write(f"| {name} | {accuracy:.2f}% | {time_ms:.2f} | {fps:.1f} | {samples} |\n")
        
        f.write("\n## 跨平台对比分析\n\n")
        
        # macOS vs Android 对比
        macos_cpp = results.get('macOS C++')
        android_cpp = results.get('Android C++')
        
        if macos_cpp and android_cpp:
            f.write("### macOS vs Android C++ 统一版本\n\n")
            
            macos_time = macos_cpp.get('average_inference_time_ms', 0)
            android_time = android_cpp.get('average_inference_time_ms', 0)
            
            if macos_time > 0 and android_time > 0:
                ratio = macos_time / android_time
                f.write(f"- macOS 推理时间: {macos_time:.2f} ms\n")
                f.write(f"- Android 推理时间: {android_time:.2f} ms\n")
                f.write(f"- 性能比 (macOS/Android): {ratio:.2f}x\n\n")
        
        # C vs C++ 对比
        macos_c = results.get('macOS C')
        macos_cpp = results.get('macOS C++')
        
        if macos_c and macos_cpp:
            f.write("### macOS C vs C++ 统一版本\n\n")
            
            c_time = macos_c.get('average_inference_time_ms', 0)
            cpp_time = macos_cpp.get('average_inference_time_ms', 0)
            
            if c_time > 0 and cpp_time > 0:
                f.write(f"- C 版本推理时间: {c_time:.2f} ms\n")
                f.write(f"- C++ 版本推理时间: {cpp_time:.2f} ms\n")
                f.write(f"- 性能差异: {abs(c_time - cpp_time):.2f} ms\n\n")
        
        f.write("## 统一版本优势\n\n")
        f.write("1. **代码维护**: 单一源码支持多平台，减少维护成本\n")
        f.write("2. **一致性**: 相同的算法逻辑保证结果一致性\n")
        f.write("3. **可移植性**: 通过预处理器宏轻松适配新平台\n")
        f.write("4. **性能**: 在不同平台上都能获得良好的推理性能\n\n")
        
        f.write("## 结论\n\n")
        f.write("统一版本代码成功实现了跨平台部署，在macOS和Android平台上都能正常运行MNIST推理任务。")
        f.write("不同平台的性能差异主要来自硬件性能和系统优化的不同。")
        f.write("统一版本的设计为AI模型的跨平台部署提供了一个高效的解决方案。\n")
    
    print(f"✓ 统一版本分析报告已生成: {report_file}")

if __name__ == "__main__":
    generate_unified_analysis()
EOF

python unified_performance_analysis.py
if [ $? -eq 0 ]; then
    print_success "统一版本性能分析完成"
else
    print_error "统一版本性能分析失败"
fi

print_step "结果展示"
echo ""
echo "🎉 === 统一版本跨平台测试完成！==="
echo ""

# 检查生成的结果文件
RESULTS_DIR="results"
TOTAL_CONFIGS=0

echo "📊 统一版本测试结果汇总:"

if [ -f "$RESULTS_DIR/python_inference_mnist_results.json" ]; then
    echo "  ✓ Python版本结果"
    ((TOTAL_CONFIGS++))
fi

if [ -f "$RESULTS_DIR/macos_unified_cpp_results.txt" ]; then
    echo "  ✓ macOS C++ 统一版本结果"
    ((TOTAL_CONFIGS++))
fi

if [ -f "$RESULTS_DIR/macos_unified_c_results.txt" ]; then
    echo "  ✓ macOS C 统一版本结果"
    ((TOTAL_CONFIGS++))
fi

if [ -f "$RESULTS_DIR/android_unified_cpp_results.txt" ]; then
    echo "  ✓ Android C++ 统一版本结果"
    ((TOTAL_CONFIGS++))
fi

if [ -f "$RESULTS_DIR/android_unified_c_results.txt" ]; then
    echo "  ✓ Android C 统一版本结果"
    ((TOTAL_CONFIGS++))
fi

echo ""
echo "📈 成功完成 $TOTAL_CONFIGS/5 个配置的测试"

if [ -f "$RESULTS_DIR/unified_cross_platform_report.md" ]; then
    echo "📋 统一版本详细报告: $RESULTS_DIR/unified_cross_platform_report.md"
fi

if [ -f "$RESULTS_DIR/unified_cross_platform_analysis.png" ]; then
    echo "📊 统一版本可视化图表: $RESULTS_DIR/unified_cross_platform_analysis.png"
    echo ""
    echo "🖼️  查看图表 (macOS): open $RESULTS_DIR/unified_cross_platform_analysis.png"
fi

if [ -f "$RESULTS_DIR/unified_deployment_report.md" ]; then
    echo "📄 部署报告: $RESULTS_DIR/unified_deployment_report.md"
fi

echo ""
echo "💡 如需重新运行特定测试:"
echo "   - Python: python inference/python_inference_mnist.py"
echo "   - macOS 统一版本: cd build && ./deploy_and_test_unified.sh macos"
echo "   - Android 统一版本: cd build && ./deploy_and_test_unified.sh android"
echo "   - 性能分析: python unified_performance_analysis.py"
echo ""

# 显示统一版本优势
echo "🌟 统一版本优势:"
echo "   ✅ 单一源码支持多平台"
echo "   ✅ 降低代码维护成本"
echo "   ✅ 保证跨平台一致性"
echo "   ✅ 便于新平台适配"
echo ""

print_success "统一版本跨平台MNIST推理测试流程完成！"

# 清理临时文件
rm -f unified_performance_analysis.py 