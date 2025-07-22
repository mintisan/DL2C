#!/usr/bin/env python3
"""
Android跨平台MNIST推理性能分析工具
比较本地(Python/C/C++) vs Android(C/C++)设备的推理性能
支持完整的多语言、跨平台性能对比分析
"""

import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

class CrossPlatformAnalyzer:
    def __init__(self):
        self.results_dir = "results"
        self.local_results = {}
        self.android_results = {}
        
    def load_local_results(self):
        """加载本地所有语言版本的推理结果"""
        local_files = {
            "Python": "python_inference_mnist_results.json",
            "C": "c_inference_mnist_results.json", 
            "C++": "cpp_inference_mnist_results.json"
        }
        
        for lang, filename in local_files.items():
            filepath = f"{self.results_dir}/{filename}"
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.local_results[lang] = self.extract_summary_from_json(data, lang)
                print(f"✓ 已加载本地{lang}推理结果")
            else:
                print(f"❌ 本地{lang}推理结果文件不存在: {filepath}")
                
    def extract_summary_from_json(self, data, language):
        """从JSON数据中提取汇总信息"""
        results = data.get("results", [])
        if not results:
            return {}
            
        # 计算汇总统计信息
        total_samples = len(results)
        correct_predictions = sum(1 for r in results if r.get("is_correct", False))
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        inference_times = [r.get("inference_time_ms", 0) for r in results]
        avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        
        return {
            "platform": f"macOS Apple Silicon ({language})",
            "framework": data.get("framework", f"ONNX Runtime {language} API"),
            "summary": {
                "total_samples": total_samples,
                "correct_predictions": correct_predictions,
                "accuracy": accuracy,
                "average_inference_time_ms": avg_time,
                "fps": fps,
                "min_time": min(inference_times) if inference_times else 0,
                "max_time": max(inference_times) if inference_times else 0,
                "std_time": np.std(inference_times) if inference_times else 0
            },
            "detailed_results": results
        }
        
    def load_android_results(self):
        """加载Android C和C++推理结果"""
        android_files = {
            "C++": "android_real_onnx_results.txt",
            "C": "android_real_onnx_c_results.txt"
        }
        
        for lang, filename in android_files.items():
            filepath = f"{self.results_dir}/{filename}"
            if os.path.exists(filepath):
                self.android_results[lang] = self.parse_android_results(filepath, lang)
                print(f"✓ 已加载Android {lang}推理结果")
            else:
                print(f"❌ Android {lang}推理结果文件不存在: {filepath}")
                
    def parse_android_results(self, filename, language):
        """解析Android推理结果文件"""
        results = {
            "platform": f"Android ARM64 ({language})",
            "framework": f"ONNX Runtime {language} API (Android)",
            "summary": {}
        }
        
        try:
            with open(filename, 'r') as f:
                content = f.read()
                
            # 解析关键指标
            lines = content.split('\n')
            for line in lines:
                if "准确率:" in line:
                    accuracy = float(line.split(':')[1].strip().replace('%', '')) / 100.0
                    results["summary"]["accuracy"] = accuracy
                elif "平均推理时间:" in line:
                    time_ms = float(line.split(':')[1].strip().split()[0])
                    results["summary"]["average_inference_time_ms"] = time_ms
                elif "推理 FPS:" in line:
                    fps = float(line.split(':')[1].strip())
                    results["summary"]["fps"] = fps
                elif "测试样本数:" in line:
                    samples = int(line.split(':')[1].strip())
                    results["summary"]["total_samples"] = samples
                    
            # 计算正确预测数（基于准确率和样本数）
            total_samples = results["summary"].get("total_samples", 0)
            accuracy = results["summary"].get("accuracy", 0)
            results["summary"]["correct_predictions"] = int(total_samples * accuracy)
                    
        except Exception as e:
            print(f"解析Android {language}结果时出错: {e}")
            
        return results
        
    def generate_comprehensive_report(self):
        """生成全面的跨平台性能对比报告"""
        if not self.local_results and not self.android_results:
            print("缺少性能数据，无法生成对比报告")
            return
            
        report = f"""# 跨平台 MNIST 推理性能全面对比报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 测试环境概述

### 本地环境 (macOS Apple Silicon)
- **硬件**: Apple Silicon M系列处理器
- **架构**: arm64 (64位)
- **操作系统**: macOS
- **测试语言**: Python, C, C++

### Android环境
- **硬件**: ARM64移动处理器
- **架构**: arm64-v8a
- **操作系统**: Android 15
- **设备型号**: 24129PN74C
- **测试语言**: C, C++

## 🔥 核心性能对比表

| 平台/语言 | 准确率 | 推理时间(ms) | FPS | 样本数 | 框架 |
|-----------|--------|-------------|-----|--------|------|
"""
        
        # 添加本地结果
        for lang, data in self.local_results.items():
            summary = data.get("summary", {})
            framework = data.get("framework", "")
            report += f"| macOS {lang} | {summary.get('accuracy', 0):.2%} | "
            report += f"{summary.get('average_inference_time_ms', 0):.3f} | "
            report += f"{summary.get('fps', 0):.1f} | "
            report += f"{summary.get('total_samples', 0)} | "
            report += f"{framework} |\n"
            
        # 添加Android结果
        for lang, data in self.android_results.items():
            summary = data.get("summary", {})
            framework = data.get("framework", "")
            report += f"| Android {lang} | {summary.get('accuracy', 0):.2%} | "
            report += f"{summary.get('average_inference_time_ms', 0):.3f} | "
            report += f"{summary.get('fps', 0):.1f} | "
            report += f"{summary.get('total_samples', 0)} | "
            report += f"{framework} |\n"
            
        # 性能分析部分
        report += "\n## ⚡ 性能深度分析\n\n"
        
        # 找到最快和最慢的配置
        all_configs = {}
        for lang, data in self.local_results.items():
            key = f"macOS {lang}"
            all_configs[key] = data.get("summary", {})
        for lang, data in self.android_results.items():
            key = f"Android {lang}"
            all_configs[key] = data.get("summary", {})
            
        if all_configs:
            # 按FPS排序
            sorted_by_fps = sorted(all_configs.items(), 
                                 key=lambda x: x[1].get('fps', 0), reverse=True)
            fastest = sorted_by_fps[0]
            slowest = sorted_by_fps[-1]
            
            # 按推理时间排序  
            sorted_by_time = sorted(all_configs.items(),
                                  key=lambda x: x[1].get('average_inference_time_ms', float('inf')))
            fastest_time = sorted_by_time[0]
            slowest_time = sorted_by_time[-1]
            
            report += f"""### 🚀 **推理速度排行榜**

1. **🥇 最快配置**: {fastest[0]}
   - FPS: {fastest[1].get('fps', 0):.1f}
   - 推理时间: {fastest[1].get('average_inference_time_ms', 0):.3f}ms
   - 准确率: {fastest[1].get('accuracy', 0):.2%}

2. **🐌 最慢配置**: {slowest[0]}
   - FPS: {slowest[1].get('fps', 0):.1f}
   - 推理时间: {slowest[1].get('average_inference_time_ms', 0):.3f}ms
   - 准确率: {slowest[1].get('accuracy', 0):.2%}

**性能倍数差距**: {fastest[1].get('fps', 0) / slowest[1].get('fps', 1):.1f}x

"""
            
            # 语言对比分析
            report += "### 🔤 **语言性能对比**\n\n"
            
            # 本地语言对比
            if len(self.local_results) >= 2:
                local_fps = {lang: data.get("summary", {}).get("fps", 0) 
                           for lang, data in self.local_results.items()}
                max_local = max(local_fps.items(), key=lambda x: x[1])
                min_local = min(local_fps.items(), key=lambda x: x[1])
                
                report += f"**本地性能 (macOS)**:\n"
                report += f"- 最快: {max_local[0]} ({max_local[1]:.1f} FPS)\n"
                report += f"- 最慢: {min_local[0]} ({min_local[1]:.1f} FPS)\n"
                report += f"- 性能差距: {max_local[1] / min_local[1]:.1f}x\n\n"
                
            # Android语言对比
            if len(self.android_results) >= 2:
                android_fps = {lang: data.get("summary", {}).get("fps", 0) 
                             for lang, data in self.android_results.items()}
                max_android = max(android_fps.items(), key=lambda x: x[1])
                min_android = min(android_fps.items(), key=lambda x: x[1])
                
                report += f"**Android性能**:\n"
                report += f"- 最快: {max_android[0]} ({max_android[1]:.1f} FPS)\n"
                report += f"- 最慢: {min_android[0]} ({min_android[1]:.1f} FPS)\n"
                report += f"- 性能差距: {max_android[1] / min_android[1]:.1f}x\n\n"
                
            # 跨平台C++对比
            if "C++" in self.local_results and "C++" in self.android_results:
                local_cpp = self.local_results["C++"].get("summary", {})
                android_cpp = self.android_results["C++"].get("summary", {})
                
                fps_ratio = android_cpp.get("fps", 0) / local_cpp.get("fps", 1)
                time_ratio = android_cpp.get("average_inference_time_ms", 0) / local_cpp.get("average_inference_time_ms", 1)
                
                report += f"""### 🌐 **跨平台 C++ 性能对比**

| 指标 | macOS C++ | Android C++ | 性能比 | 分析 |
|------|-----------|-------------|--------|------|
| FPS | {local_cpp.get('fps', 0):.1f} | {android_cpp.get('fps', 0):.1f} | {fps_ratio:.3f}x | Android为macOS的{fps_ratio:.1%} |
| 推理时间 | {local_cpp.get('average_inference_time_ms', 0):.3f}ms | {android_cpp.get('average_inference_time_ms', 0):.3f}ms | {time_ratio:.3f}x | Android比macOS慢{time_ratio:.1f}倍 |
| 准确率 | {local_cpp.get('accuracy', 0):.2%} | {android_cpp.get('accuracy', 0):.2%} | {android_cpp.get('accuracy', 0) / local_cpp.get('accuracy', 1):.3f}x | 准确率一致性良好 |

"""
                
        # 算法质量分析
        report += "## 🎯 **算法质量分析**\n\n"
        
        all_accuracies = []
        for configs in [self.local_results, self.android_results]:
            for lang, data in configs.items():
                acc = data.get("summary", {}).get("accuracy", 0)
                all_accuracies.append(acc)
                
        if all_accuracies:
            avg_acc = sum(all_accuracies) / len(all_accuracies)
            min_acc = min(all_accuracies)
            max_acc = max(all_accuracies)
            
            report += f"""### ✅ **准确率统计**
- **平均准确率**: {avg_acc:.2%}
- **最高准确率**: {max_acc:.2%}
- **最低准确率**: {min_acc:.2%}
- **准确率稳定性**: {(max_acc - min_acc):.2%} 变动范围

### 🔬 **算法一致性**
跨平台和跨语言的算法实现保持了极高的一致性，准确率变动在{(max_acc - min_acc):.1%}以内，
证明ONNX模型格式的标准化和ONNX Runtime的跨平台兼容性。

"""

        # 实际应用建议
        report += """## 💡 **部署建议**

### 🚀 **性能优先场景**
- **推荐**: 本地C++或C语言版本
- **原因**: 最高的推理速度和最低的延迟
- **适用**: 实时处理、高并发场景

### 📱 **移动端部署**
- **推荐**: Android C版本
- **原因**: 相对较好的性能和更小的内存占用
- **适用**: 手机App、嵌入式设备

### 🐍 **开发原型**
- **推荐**: Python版本
- **原因**: 开发效率高，便于调试和修改
- **适用**: 算法验证、快速原型开发

### ⚖️ **平衡选择**
- **推荐**: 本地C++开发 + Android C部署
- **原因**: 开发阶段效率高，部署阶段性能优
- **适用**: 商业产品开发

---
*报告生成工具: DL2C 跨平台性能分析系统 v2.0*
"""
        
        # 保存报告
        report_file = f"{self.results_dir}/comprehensive_cross_platform_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(f"✓ 全面跨平台性能对比报告已生成: {report_file}")
        return report_file
        
    def create_comprehensive_charts(self):
        """创建全面的性能对比图表"""
        if not self.local_results and not self.android_results:
            return
            
        # 准备数据
        platforms = []
        fps_data = []
        time_data = []
        acc_data = []
        colors = []
        
        # 本地数据 (蓝色系)
        local_colors = ['#1f77b4', '#2ca02c', '#d62728']  # 蓝、绿、红
        for i, (lang, data) in enumerate(self.local_results.items()):
            summary = data.get("summary", {})
            platforms.append(f'macOS\n{lang}')
            fps_data.append(summary.get("fps", 0))
            time_data.append(summary.get("average_inference_time_ms", 0))
            acc_data.append(summary.get("accuracy", 0) * 100)
            colors.append(local_colors[i % len(local_colors)])
            
        # Android数据 (橙色系)  
        android_colors = ['#ff7f0e', '#ffbb78']  # 橙色、浅橙色
        for i, (lang, data) in enumerate(self.android_results.items()):
            summary = data.get("summary", {})
            platforms.append(f'Android\n{lang}')
            fps_data.append(summary.get("fps", 0))
            time_data.append(summary.get("average_inference_time_ms", 0))
            acc_data.append(summary.get("accuracy", 0) * 100)
            colors.append(android_colors[i % len(android_colors)])
        
        # 创建综合对比图
        fig = plt.figure(figsize=(18, 12))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. FPS性能对比 (左上)
        ax1 = plt.subplot(2, 3, 1)
        bars1 = ax1.bar(platforms, fps_data, color=colors)
        ax1.set_title('推理性能对比 (FPS)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('FPS (frames per second)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{fps_data[i]:.0f}', ha='center', va='bottom', fontweight='bold')
                    
        # 2. 推理时间对比 (右上)
        ax2 = plt.subplot(2, 3, 2)
        bars2 = ax2.bar(platforms, time_data, color=colors)
        ax2.set_title('平均推理时间对比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('时间 (ms)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{time_data[i]:.3f}', ha='center', va='bottom', fontweight='bold')
                    
        # 3. 准确率对比 (中上)
        ax3 = plt.subplot(2, 3, 3)
        bars3 = ax3.bar(platforms, acc_data, color=colors)
        ax3.set_title('模型准确率对比', fontsize=14, fontweight='bold')
        ax3.set_ylabel('准确率 (%)', fontsize=12)
        ax3.set_ylim(95, 100)  # 聚焦高准确率区间
        ax3.tick_params(axis='x', rotation=45)
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{acc_data[i]:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. 性能效率对比 (左下) - FPS vs 推理时间
        ax4 = plt.subplot(2, 3, 4)
        scatter = ax4.scatter(time_data, fps_data, c=colors, s=100, alpha=0.7)
        ax4.set_xlabel('推理时间 (ms)', fontsize=12)
        ax4.set_ylabel('FPS', fontsize=12)
        ax4.set_title('性能效率分布', fontsize=14, fontweight='bold')
        for i, platform in enumerate(platforms):
            ax4.annotate(platform.replace('\n', ' '), 
                        (time_data[i], fps_data[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 5. 跨平台性能对比雷达图 (右下)
        if len(fps_data) >= 2:
            ax5 = plt.subplot(2, 3, 5, projection='polar')
            
            # 标准化数据用于雷达图
            max_fps = max(fps_data) if fps_data else 1
            max_acc = max(acc_data) if acc_data else 1
            min_time = min(time_data) if time_data else 1
            
            # 雷达图数据 (越大越好，所以推理时间要反转)
            radar_data = []
            for i in range(len(platforms)):
                fps_norm = fps_data[i] / max_fps
                acc_norm = acc_data[i] / max_acc
                time_norm = min_time / time_data[i] if time_data[i] > 0 else 0  # 时间越小越好
                radar_data.append([fps_norm, acc_norm, time_norm])
            
            # 设置雷达图
            categories = ['推理速度', '准确率', '时间效率']
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            
            for i, data in enumerate(radar_data):
                values = data + data[:1]  # 闭合
                ax5.plot(angles, values, 'o-', linewidth=2, label=platforms[i], color=colors[i])
                ax5.fill(angles, values, alpha=0.25, color=colors[i])
            
            ax5.set_xticks(angles[:-1])
            ax5.set_xticklabels(categories)
            ax5.set_ylim(0, 1)
            ax5.set_title('综合性能雷达图', fontsize=14, fontweight='bold', pad=20)
            ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 6. 性能总结图表 (中下)
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # 性能排行文本
        performance_text = "🏆 性能排行榜\n\n"
        
        # 按FPS排序
        sorted_indices = sorted(range(len(fps_data)), key=lambda i: fps_data[i], reverse=True)
        medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
        
        for rank, idx in enumerate(sorted_indices):
            medal = medals[rank] if rank < len(medals) else f"{rank+1}️⃣"
            performance_text += f"{medal} {platforms[idx].replace(chr(10), ' ')}\n"
            performance_text += f"   FPS: {fps_data[idx]:.1f}\n"
            performance_text += f"   时间: {time_data[idx]:.3f}ms\n"
            performance_text += f"   准确率: {acc_data[idx]:.1f}%\n\n"
        
        ax6.text(0.1, 0.9, performance_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        chart_file = f"{self.results_dir}/comprehensive_cross_platform_analysis.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 全面性能对比图表已生成: {chart_file}")
        return chart_file
        
    def run_comprehensive_analysis(self):
        """运行完整的跨平台分析"""
        print("=== 跨平台 MNIST 推理性能全面分析 ===")
        print("📊 正在加载所有平台和语言的推理结果...")
        
        self.load_local_results()
        self.load_android_results()
        
        total_configs = len(self.local_results) + len(self.android_results)
        
        if total_configs == 0:
            print("\n❌ 未找到任何推理结果文件")
            print("请先运行以下测试：")
            print("  - 本地: python inference/python_inference_mnist.py")
            print("  - 本地: inference/cpp_inference_mnist")  
            print("  - 本地: inference/c_inference_mnist")
            print("  - Android: ./build/deploy_and_test_real_onnx.sh")
            return
            
        print(f"\n📈 发现 {total_configs} 个配置的测试结果")
        print(f"   - 本地配置: {len(self.local_results)} 个")
        print(f"   - Android配置: {len(self.android_results)} 个")
        
        if total_configs >= 2:
            report_file = self.generate_comprehensive_report()
            chart_file = self.create_comprehensive_charts()
            
            print(f"\n🎉 跨平台全面分析完成！")
            print(f"📋 详细报告: {report_file}")
            print(f"📊 可视化图表: {chart_file}")
            print(f"\n💡 发现 {total_configs} 个配置，可进行全面性能对比分析")
        else:
            print(f"\n⏳ 配置数量不足 (当前{total_configs}个，需要至少2个)")
            print("请完成更多平台和语言的推理测试以获得有意义的对比结果")

if __name__ == "__main__":
    analyzer = CrossPlatformAnalyzer()
    analyzer.run_comprehensive_analysis() 