#!/usr/bin/env python3
"""
三种语言推理性能对比分析脚本
对比Python、C++、C语言的ONNX Runtime推理性能
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class LanguagePerformanceComparator:
    def __init__(self, results_dir="./results"):
        self.results_dir = Path(results_dir)
        self.python_data = None
        self.cpp_data = None
        self.c_data = None
        
    def load_results(self):
        """加载三种语言的推理结果"""
        print("🔍 加载推理结果文件...")
        
        # Python结果
        python_path = self.results_dir / "python_inference_results.json"
        if python_path.exists():
            with open(python_path, 'r', encoding='utf-8') as f:
                self.python_data = json.load(f)
            print(f"✓ Python结果: {python_path}")
        else:
            print(f"✗ Python结果文件不存在: {python_path}")
            
        # C++结果
        cpp_path = self.results_dir / "cpp_inference_results.json"
        if cpp_path.exists():
            with open(cpp_path, 'r') as f:
                self.cpp_data = json.load(f)
            print(f"✓ C++结果: {cpp_path}")
        else:
            print(f"✗ C++结果文件不存在: {cpp_path}")
            
        # C结果
        c_path = self.results_dir / "c_inference_results.json"
        if c_path.exists():
            with open(c_path, 'r') as f:
                self.c_data = json.load(f)
            print(f"✓ C结果: {c_path}")
        else:
            print(f"✗ C结果文件不存在: {c_path}")
            
    def analyze_performance(self):
        """分析性能数据"""
        print("\n📊 性能分析报告")
        print("=" * 80)
        
        if not any([self.python_data, self.cpp_data, self.c_data]):
            print("❌ 没有找到任何结果文件，请先运行推理测试")
            return
            
        # 基础信息表格
        print("\n🎯 基础性能指标:")
        print("-" * 80)
        print(f"{'语言':<10} {'框架':<25} {'平均时间(ms)':<12} {'FPS':<8} {'准确率':<8}")
        print("-" * 80)
        
        times = {}
        
        if self.python_data:
            summary = self.python_data['summary']
            accuracy = f"{summary['accuracy']:.2%}" if 'accuracy' in summary else "N/A"
            print(f"{'Python':<10} {'ONNX Runtime Python':<25} "
                  f"{summary['average_inference_time_ms']:<12.2f} "
                  f"{summary['fps']:<8.1f} {accuracy:<8}")
            times['Python'] = summary['average_inference_time_ms']
            
        if self.cpp_data:
            summary = self.cpp_data['summary']
            print(f"{'C++':<10} {'ONNX Runtime C++':<25} "
                  f"{summary['average_inference_time_ms']:<12.2f} "
                  f"{summary['fps']:<8.1f} {'N/A':<8}")
            times['C++'] = summary['average_inference_time_ms']
            
        if self.c_data:
            summary = self.c_data['summary']
            print(f"{'C':<10} {'ONNX Runtime C API':<25} "
                  f"{summary['average_inference_time_ms']:<12.2f} "
                  f"{summary['fps']:<8.1f} {'N/A':<8}")
            times['C'] = summary['average_inference_time_ms']
            
        # 性能对比分析
        if len(times) >= 2:
            print(f"\n🚀 性能提升分析:")
            print("-" * 50)
            
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
                    
        # 详细统计分析
        self.detailed_analysis()
        
    def detailed_analysis(self):
        """详细统计分析"""
        print(f"\n📈 详细统计分析:")
        print("-" * 50)
        
        # 分析每个语言的推理时间分布
        for name, data in [("Python", self.python_data), ("C++", self.cpp_data), ("C", self.c_data)]:
            if data and 'results' in data:
                times = [r['inference_time_ms'] for r in data['results'] if 'inference_time_ms' in r]
                if times:
                    times_array = np.array(times)
                    print(f"\n{name} 推理时间统计:")
                    print(f"  平均时间: {np.mean(times_array):.2f} ms")
                    print(f"  标准差:   {np.std(times_array):.2f} ms")
                    print(f"  最小时间: {np.min(times_array):.2f} ms")
                    print(f"  最大时间: {np.max(times_array):.2f} ms")
                    print(f"  中位数:   {np.median(times_array):.2f} ms")
                    
    def generate_visualization(self):
        """生成可视化图表"""
        try:
            print(f"\n📊 生成性能对比图表...")
            
            # 准备数据
            languages = []
            avg_times = []
            fps_values = []
            
            for name, data in [("Python", self.python_data), ("C++", self.cpp_data), ("C", self.c_data)]:
                if data and 'summary' in data:
                    languages.append(name)
                    avg_times.append(data['summary']['average_inference_time_ms'])
                    fps_values.append(data['summary']['fps'])
                    
            if len(languages) < 2:
                print("⚠️  数据不足，跳过图表生成")
                return
                
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 推理时间对比
            colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(languages)]
            bars1 = ax1.bar(languages, avg_times, color=colors, alpha=0.8)
            ax1.set_ylabel('推理时间 (ms)')
            ax1.set_title('平均推理时间对比')
            ax1.set_ylim(0, max(avg_times) * 1.2)
            
            # 在柱状图上添加数值标签
            for bar, time in zip(bars1, avg_times):
                height = bar.get_height()
                ax1.annotate(f'{time:.2f}ms',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            # FPS对比
            bars2 = ax2.bar(languages, fps_values, color=colors, alpha=0.8)
            ax2.set_ylabel('推理速度 (FPS)')
            ax2.set_title('推理速度对比')
            ax2.set_ylim(0, max(fps_values) * 1.2)
            
            # 在柱状图上添加数值标签
            for bar, fps in zip(bars2, fps_values):
                height = bar.get_height()
                ax2.annotate(f'{fps:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = self.results_dir / "performance_comparison.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✓ 图表已保存: {chart_path}")
            
        except ImportError:
            print("⚠️  matplotlib未安装，跳过图表生成")
            print("可以运行 'pip install matplotlib' 来安装")
        except Exception as e:
            print(f"⚠️  图表生成失败: {e}")
            
    def generate_report(self):
        """生成详细的报告文件"""
        print(f"\n📝 生成详细报告...")
        
        report_path = self.results_dir / "performance_comparison_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# MNIST模型三种语言推理性能对比报告\n\n")
            f.write(f"生成时间: {self.get_timestamp()}\n\n")
            
            # 概要
            f.write("## 概要\n\n")
            f.write("本报告对比了使用ONNX Runtime在Python、C++和C语言下的MNIST模型推理性能。\n\n")
            
            # 测试环境
            f.write("## 测试环境\n\n")
            f.write("- 操作系统: macOS\n")
            f.write("- ONNX Runtime版本: 1.16.0\n")
            f.write("- 模型: MNIST CNN (PyTorch训练)\n")
            f.write("- 测试数据: 随机生成的28x28图像\n\n")
            
            # 性能结果表格
            f.write("## 性能测试结果\n\n")
            f.write("| 语言 | 框架 | 平均推理时间(ms) | FPS | 准确率 |\n")
            f.write("|------|------|------------------|-----|--------|\n")
            
            for name, data in [("Python", self.python_data), ("C++", self.cpp_data), ("C", self.c_data)]:
                if data and 'summary' in data:
                    summary = data['summary']
                    accuracy = f"{summary['accuracy']:.2%}" if 'accuracy' in summary else "N/A"
                    framework = {
                        "Python": "ONNX Runtime Python API",
                        "C++": "ONNX Runtime C++ API", 
                        "C": "ONNX Runtime C API"
                    }[name]
                    
                    f.write(f"| {name} | {framework} | "
                           f"{summary['average_inference_time_ms']:.2f} | "
                           f"{summary['fps']:.1f} | {accuracy} |\n")
            
            # 性能分析
            f.write("\n## 性能分析\n\n")
            
            times = {}
            for name, data in [("Python", self.python_data), ("C++", self.cpp_data), ("C", self.c_data)]:
                if data and 'summary' in data:
                    times[name] = data['summary']['average_inference_time_ms']
            
            if 'Python' in times:
                baseline = times['Python']
                f.write(f"### 相对于Python的性能提升\n\n")
                
                if 'C++' in times:
                    speedup = baseline / times['C++']
                    f.write(f"- **C++**: {speedup:.2f}x 加速\n")
                    
                if 'C' in times:
                    speedup = baseline / times['C']
                    f.write(f"- **C语言**: {speedup:.2f}x 加速\n")
                    
            if 'C++' in times and 'C' in times:
                f.write(f"\n### C vs C++\n\n")
                ratio = times['C++'] / times['C']
                if ratio > 1:
                    f.write(f"- C语言比C++快 {ratio:.2f}x\n")
                else:
                    f.write(f"- C++比C语言快 {1/ratio:.2f}x\n")
            
            # 结论
            f.write(f"\n## 结论\n\n")
            f.write("1. **兼容性**: C语言提供最好的跨平台兼容性\n")
            f.write("2. **性能**: 编译型语言(C/C++)显著优于解释型语言(Python)\n") 
            f.write("3. **开发效率**: Python开发最快，C语言需要更多内存管理\n")
            f.write("4. **生产部署**: 推荐使用C/C++版本进行移动端部署\n\n")
            
        print(f"✓ 详细报告已保存: {report_path}")
        
    def get_timestamp(self):
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def run_comparison(self):
        """运行完整的对比分析"""
        print("🎯 MNIST模型三种语言推理性能对比分析")
        print("=" * 60)
        
        # 加载结果
        self.load_results()
        
        # 分析性能
        self.analyze_performance()
        
        # 生成可视化
        self.generate_visualization()
        
        # 生成报告
        self.generate_report()
        
        print(f"\n🎉 对比分析完成!")
        print("📁 生成的文件:")
        print(f"  - 性能图表: {self.results_dir}/performance_comparison.png")
        print(f"  - 详细报告: {self.results_dir}/performance_comparison_report.md")

def main():
    """主函数"""
    # 检查结果目录
    results_dir = "./results"
    if not os.path.exists(results_dir):
        print(f"❌ 结果目录不存在: {results_dir}")
        print("请先运行推理测试生成结果文件")
        sys.exit(1)
        
    # 运行对比分析
    comparator = LanguagePerformanceComparator(results_dir)
    comparator.run_comparison()

if __name__ == "__main__":
    main() 