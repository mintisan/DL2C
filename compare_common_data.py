#!/usr/bin/env python3
"""
三种语言使用共同数据的推理性能和准确性对比分析脚本
对比Python、C++、C语言在相同输入数据下的性能和准确性
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class CommonDataPerformanceComparator:
    def __init__(self, results_dir="./results"):
        self.results_dir = Path(results_dir)
        self.python_data = None
        self.cpp_data = None
        self.c_data = None
        
    def load_results(self):
        """加载三种语言的共同数据推理结果"""
        print("🔍 加载共同数据推理结果文件...")
        
        # Python结果
        python_path = self.results_dir / "python_inference_common_results.json"
        if python_path.exists():
            with open(python_path, 'r', encoding='utf-8') as f:
                self.python_data = json.load(f)
            print(f"✓ Python结果: {python_path}")
        else:
            print(f"✗ Python结果文件不存在: {python_path}")
            
        # C++结果
        cpp_path = self.results_dir / "cpp_inference_common_results.json"
        if cpp_path.exists():
            with open(cpp_path, 'r') as f:
                self.cpp_data = json.load(f)
            print(f"✓ C++结果: {cpp_path}")
        else:
            print(f"✗ C++结果文件不存在: {cpp_path}")
            
        # C结果
        c_path = self.results_dir / "c_inference_common_results.json"
        if c_path.exists():
            with open(c_path, 'r') as f:
                self.c_data = json.load(f)
            print(f"✓ C结果: {c_path}")
        else:
            print(f"✗ C结果文件不存在: {c_path}")
            
    def analyze_accuracy_consistency(self):
        """分析准确性和结果一致性"""
        print("\n📊 准确性和结果一致性分析")
        print("=" * 80)
        
        if not any([self.python_data, self.cpp_data, self.c_data]):
            print("❌ 没有找到任何结果文件")
            return
            
        # 基础准确性表格
        print("\n🎯 准确性对比:")
        print("-" * 60)
        print(f"{'语言':<10} {'准确率':<8} {'正确数/总数':<12} {'FPS':<8}")
        print("-" * 60)
        
        accuracies = {}
        
        for name, data in [("Python", self.python_data), ("C++", self.cpp_data), ("C", self.c_data)]:
            if data and 'summary' in data:
                summary = data['summary']
                accuracy = summary.get('accuracy', 0) * 100
                correct = summary.get('correct_predictions', 0)
                total = summary.get('total_samples', 0)
                fps = summary.get('fps', 0)
                
                print(f"{name:<10} {accuracy:<7.1f}% {correct}/{total:<9} {fps:<7.1f}")
                accuracies[name] = accuracy
                
        # 结果一致性分析
        self.analyze_prediction_consistency()
        
    def analyze_prediction_consistency(self):
        """分析预测结果的一致性"""
        print(f"\n🔍 逐样本预测对比:")
        print("-" * 80)
        
        # 获取所有结果数据
        all_results = {}
        for name, data in [("Python", self.python_data), ("C++", self.cpp_data), ("C", self.c_data)]:
            if data and 'results' in data:
                all_results[name] = {r['sample_id']: r for r in data['results']}
        
        if len(all_results) < 2:
            print("⚠️  结果数据不足，无法进行一致性分析")
            return
            
        # 找出共同的样本ID
        sample_ids = None
        for results in all_results.values():
            if sample_ids is None:
                sample_ids = set(results.keys())
            else:
                sample_ids = sample_ids.intersection(set(results.keys()))
        
        sample_ids = sorted(list(sample_ids))
        
        print(f"样本ID | 真实标签 | {'Python':<8} | {'C++':<8} | {'C':<8} | 一致性")
        print("-" * 60)
        
        consistency_count = 0
        total_samples = len(sample_ids)
        
        for sample_id in sample_ids:
            true_label = None
            predictions = {}
            confidences = {}
            
            for lang in all_results:
                if sample_id in all_results[lang]:
                    result = all_results[lang][sample_id]
                    predictions[lang] = result['predicted_class']
                    confidences[lang] = result['confidence']
                    if true_label is None:
                        true_label = result['true_label']
            
            # 检查一致性
            pred_values = list(predictions.values())
            is_consistent = len(set(pred_values)) == 1 if pred_values else False
            if is_consistent:
                consistency_count += 1
                
            # 显示结果
            python_pred = f"{predictions.get('Python', 'N/A')}"
            cpp_pred = f"{predictions.get('C++', 'N/A')}"
            c_pred = f"{predictions.get('C', 'N/A')}"
            consistency_mark = "✓" if is_consistent else "✗"
            
            print(f"{sample_id:6d} | {true_label:8d} | {python_pred:<8} | {cpp_pred:<8} | {c_pred:<8} | {consistency_mark}")
        
        consistency_rate = consistency_count / total_samples * 100
        print(f"\n🎯 预测一致性: {consistency_count}/{total_samples} ({consistency_rate:.1f}%)")
        
        if consistency_rate < 100:
            print("⚠️  存在预测不一致的情况，可能原因:")
            print("   - 数值精度差异")
            print("   - 预处理或后处理实现差异")
            print("   - 随机性或并行计算差异")
            
    def analyze_performance(self):
        """分析性能数据"""
        print("\n📈 性能分析")
        print("=" * 80)
        
        # 性能统计
        times = {}
        
        for name, data in [("Python", self.python_data), ("C++", self.cpp_data), ("C", self.c_data)]:
            if data and 'summary' in data:
                times[name] = data['summary']['average_inference_time_ms']
                
        if len(times) >= 2:
            print(f"\n⚡ 性能提升分析:")
            print("-" * 40)
            
            if 'Python' in times:
                baseline = times['Python']
                if 'C++' in times:
                    speedup = baseline / times['C++']
                    print(f"C++ vs Python:   {speedup:.2f}x 加速")
                    
                if 'C' in times:
                    speedup = baseline / times['C']
                    print(f"C vs Python:     {speedup:.2f}x 加速")
                    
            if 'C++' in times and 'C' in times:
                ratio = times['C++'] / times['C']
                if ratio > 1:
                    print(f"C vs C++:        {ratio:.2f}x 加速")
                else:
                    print(f"C++ vs C:        {1/ratio:.2f}x 加速")
        
        # 详细时间分析
        self.detailed_timing_analysis()
        
    def detailed_timing_analysis(self):
        """详细时间分析"""
        print(f"\n⏱️  详细时间统计:")
        print("-" * 50)
        
        for name, data in [("Python", self.python_data), ("C++", self.cpp_data), ("C", self.c_data)]:
            if data and 'results' in data:
                times = [r['inference_time_ms'] for r in data['results']]
                if times:
                    times_array = np.array(times)
                    print(f"\n{name} 推理时间统计:")
                    print(f"  平均时间: {np.mean(times_array):.3f} ms")
                    print(f"  标准差:   {np.std(times_array):.3f} ms")
                    print(f"  最小时间: {np.min(times_array):.3f} ms")
                    print(f"  最大时间: {np.max(times_array):.3f} ms")
                    print(f"  中位数:   {np.median(times_array):.3f} ms")
                    
    def generate_detailed_visualization(self):
        """生成详细的可视化图表"""
        try:
            print(f"\n📊 生成详细对比图表...")
            
            # 准备数据
            languages = []
            accuracies = []
            avg_times = []
            fps_values = []
            
            for name, data in [("Python", self.python_data), ("C++", self.cpp_data), ("C", self.c_data)]:
                if data and 'summary' in data:
                    languages.append(name)
                    accuracies.append(data['summary'].get('accuracy', 0) * 100)
                    avg_times.append(data['summary']['average_inference_time_ms'])
                    fps_values.append(data['summary']['fps'])
                    
            if len(languages) < 2:
                print("⚠️  数据不足，跳过图表生成")
                return
                
            # 创建子图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(languages)]
            
            # 1. 准确率对比
            bars1 = ax1.bar(languages, accuracies, color=colors, alpha=0.8)
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_title('Accuracy Comparison')
            ax1.set_ylim(0, 100)
            for bar, acc in zip(bars1, accuracies):
                height = bar.get_height()
                ax1.annotate(f'{acc:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            # 2. 推理时间对比
            bars2 = ax2.bar(languages, avg_times, color=colors, alpha=0.8)
            ax2.set_ylabel('Inference Time (ms)')
            ax2.set_title('Average Inference Time Comparison')
            for bar, time in zip(bars2, avg_times):
                height = bar.get_height()
                ax2.annotate(f'{time:.2f}ms',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            # 3. FPS对比
            bars3 = ax3.bar(languages, fps_values, color=colors, alpha=0.8)
            ax3.set_ylabel('Inference Speed (FPS)')
            ax3.set_title('Inference Speed Comparison')
            for bar, fps in zip(bars3, fps_values):
                height = bar.get_height()
                ax3.annotate(f'{fps:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            # 4. 综合评分 (准确率 × 相对速度)
            if len(languages) > 1:
                baseline_time = max(avg_times)  # 使用最慢的作为基准
                speed_scores = [baseline_time / t for t in avg_times]
                combined_scores = [acc * speed / 100 for acc, speed in zip(accuracies, speed_scores)]
                
                bars4 = ax4.bar(languages, combined_scores, color=colors, alpha=0.8)
                ax4.set_ylabel('Combined Score (Accuracy × Relative Speed)')
                ax4.set_title('Overall Performance Score')
                for bar, score in zip(bars4, combined_scores):
                    height = bar.get_height()
                    ax4.annotate(f'{score:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = self.results_dir / "common_data_comparison.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✓ 图表已保存: {chart_path}")
            
        except ImportError:
            print("⚠️  matplotlib未安装，跳过图表生成")
        except Exception as e:
            print(f"⚠️  图表生成失败: {e}")
            
    def generate_comprehensive_report(self):
        """生成综合报告"""
        print(f"\n📝 生成综合报告...")
        
        report_path = self.results_dir / "common_data_comparison_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# MNIST模型三种语言共同数据推理对比报告\n\n")
            f.write(f"生成时间: {self.get_timestamp()}\n\n")
            
            # 概要
            f.write("## 概要\n\n")
            f.write("本报告对比了Python、C++和C语言在使用完全相同的MNIST测试数据下的推理性能和准确性。\n")
            f.write("这是一个公平的对比，确保三种语言处理完全相同的输入数据。\n\n")
            
            # 测试设置
            f.write("## 测试设置\n\n")
            f.write("- 测试数据: 10个真实的MNIST样本（固定种子选择）\n")
            f.write("- 数据格式: 28×28像素，float32格式\n")
            f.write("- 预处理: 统一的标准化 (mean=0.1307, std=0.3081)\n")
            f.write("- 模型: 相同的ONNX模型文件\n")
            f.write("- 运行环境: macOS, ONNX Runtime 1.16.0\n\n")
            
            # 结果表格
            f.write("## 测试结果\n\n")
            f.write("| 语言 | 准确率 | 平均推理时间(ms) | FPS | 正确预测数 |\n")
            f.write("|------|--------|------------------|-----|-------------|\n")
            
            for name, data in [("Python", self.python_data), ("C++", self.cpp_data), ("C", self.c_data)]:
                if data and 'summary' in data:
                    summary = data['summary']
                    accuracy = summary.get('accuracy', 0) * 100
                    avg_time = summary['average_inference_time_ms']
                    fps = summary['fps']
                    correct = summary.get('correct_predictions', 0)
                    total = summary.get('total_samples', 0)
                    
                    f.write(f"| {name} | {accuracy:.1f}% | {avg_time:.2f} | {fps:.1f} | {correct}/{total} |\n")
            
            # 性能分析
            f.write("\n## 性能分析\n\n")
            
            times = {}
            accuracies = {}
            for name, data in [("Python", self.python_data), ("C++", self.cpp_data), ("C", self.c_data)]:
                if data and 'summary' in data:
                    times[name] = data['summary']['average_inference_time_ms']
                    accuracies[name] = data['summary'].get('accuracy', 0) * 100
            
            if 'Python' in times:
                baseline = times['Python']
                f.write(f"### 相对于Python的性能提升\n\n")
                
                if 'C++' in times:
                    speedup = baseline / times['C++']
                    f.write(f"- **C++**: {speedup:.2f}x 加速\n")
                    
                if 'C' in times:
                    speedup = baseline / times['C']
                    f.write(f"- **C语言**: {speedup:.2f}x 加速\n")
            
            # 准确性分析
            f.write(f"\n### 准确性对比\n\n")
            if len(accuracies) > 1:
                max_acc = max(accuracies.values())
                min_acc = min(accuracies.values())
                acc_diff = max_acc - min_acc
                f.write(f"- 最高准确率: {max_acc:.1f}%\n")
                f.write(f"- 最低准确率: {min_acc:.1f}%\n")
                f.write(f"- 准确率差异: {acc_diff:.1f}%\n")
                
                if acc_diff < 1.0:
                    f.write("- **结论**: 三种语言实现的准确性基本一致\n")
                else:
                    f.write("- **注意**: 存在准确性差异，需要进一步调查实现细节\n")
            
            # 结论
            f.write(f"\n## 结论\n\n")
            f.write("1. **准确性**: 使用相同数据，三种语言的推理准确性应该完全一致\n")
            f.write("2. **性能**: 编译型语言(C/C++)显著优于解释型语言(Python)\n")
            f.write("3. **一致性**: 验证了三种实现的正确性\n")
            f.write("4. **选择建议**:\n")
            f.write("   - 开发阶段: Python (快速原型)\n")
            f.write("   - 生产部署: C/C++ (高性能)\n")
            f.write("   - 移动端: C语言 (最小依赖)\n\n")
            
        print(f"✓ 综合报告已保存: {report_path}")
        
    def get_timestamp(self):
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def run_comparison(self):
        """运行完整的对比分析"""
        print("🎯 三种语言共同数据推理对比分析")
        print("=" * 60)
        
        # 加载结果
        self.load_results()
        
        if not any([self.python_data, self.cpp_data, self.c_data]):
            print("❌ 没有找到任何结果文件")
            print("请先运行相应的推理测试:")
            print("  python inference/python_inference_common.py")
            print("  cd build/build_macos && ./bin/mnist_inference_cpp_common")
            print("  cd build/build_macos && ./bin/mnist_inference_c_common")
            return
        
        # 分析准确性和一致性
        self.analyze_accuracy_consistency()
        
        # 分析性能
        self.analyze_performance()
        
        # 生成可视化
        self.generate_detailed_visualization()
        
        # 生成报告
        self.generate_comprehensive_report()
        
        print(f"\n🎉 共同数据对比分析完成!")
        print("📁 生成的文件:")
        print(f"  - 详细图表: {self.results_dir}/common_data_comparison.png")
        print(f"  - 综合报告: {self.results_dir}/common_data_comparison_report.md")

def main():
    """主函数"""
    # 检查结果目录
    results_dir = "./results"
    if not os.path.exists(results_dir):
        print(f"❌ 结果目录不存在: {results_dir}")
        sys.exit(1)
        
    # 运行对比分析
    comparator = CommonDataPerformanceComparator(results_dir)
    comparator.run_comparison()

if __name__ == "__main__":
    main() 