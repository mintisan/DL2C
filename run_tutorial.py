#!/usr/bin/env python3
"""
MNIST模型部署教学脚本 - 统一版本跨平台教程
一步一步引导用户完成从训练到跨平台部署的完整流程
支持 Python + macOS C/C++ + Android C/C++ 的统一版本部署
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

class MNISTTutorial:
    def __init__(self):
        self.project_root = Path.cwd()
        self.steps_completed = []
        self.platforms_available = self.check_platform_support()
        
    def check_platform_support(self):
        """检查平台支持情况"""
        platforms = {
            'macos': True,  # 当前运行平台
            'android': False,
            'onnxruntime': False
        }
        
        # 检查 Android 支持 (adb + NDK)
        try:
            subprocess.run(["adb", "version"], capture_output=True, check=True)
            ndk_path = os.environ.get('ANDROID_NDK_ROOT') or "/opt/homebrew/share/android-ndk"
            if os.path.exists(ndk_path):
                platforms['android'] = True
        except:
            pass
        
        # 检查 ONNX Runtime
        try:
            if os.path.exists("/opt/homebrew/include/onnxruntime"):
                platforms['onnxruntime'] = True
        except:
            pass
            
        return platforms
        
    def print_step(self, step_num, title, description):
        """打印步骤标题"""
        print("\n" + "="*60)
        print(f"步骤 {step_num}: {title}")
        print("="*60)
        print(description)
        print()
        
    def wait_for_user(self, message="按回车键继续..."):
        """等待用户确认"""
        input(f"{message}")
        
    def check_dependencies(self):
        """检查Python依赖"""
        required_packages = [
            'torch', 'torchvision', 'onnx', 'onnxruntime', 
            'numpy', 'matplotlib', 'Pillow'
        ]
        
        print("检查Python依赖包...")
        missing_packages = []
        failed_packages = []
        
        for package in required_packages:
            try:
                if package == 'Pillow':
                    __import__('PIL')  # Pillow的导入名是PIL
                else:
                    __import__(package)
                print(f"✓ {package}")
            except ImportError as e:
                print(f"✗ {package} (缺失)")
                missing_packages.append(package)
            except Exception as e:
                print(f"⚠️ {package} (导入错误: {str(e)[:50]}...)")
                failed_packages.append(package)
        
        if missing_packages:
            print(f"\n缺失的包: {', '.join(missing_packages)}")
            print("请安装缺失的包:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
            
        if failed_packages:
            print(f"\n导入失败的包: {', '.join(failed_packages)}")
            print("这些包已安装但导入失败，可能是版本兼容性问题。")
            print("建议的解决方案:")
            if 'torchvision' in failed_packages:
                print("1. pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0")
            print("2. 创建新的conda环境: conda create -n mnist_deploy python=3.9")
            
            choice = input("是否继续运行教程？(y/n): ").strip().lower()
            return choice in ['y', 'yes']
        
        print("✓ 所有Python依赖包已安装")
        return True
        
    def step1_train_model(self):
        """步骤1: 训练MNIST模型"""
        self.print_step(1, "训练MNIST模型", 
                       "我们将训练一个简单的CNN模型来识别MNIST手写数字。\n"
                       "这个模型包含2个卷积层和2个全连接层。\n"
                       "训练将使用MNIST数据集，包含60000个训练样本。")
        
        # 检查模型是否已存在
        model_path = self.project_root / "models" / "mnist_model.pth"
        if model_path.exists():
            print(f"✓ 发现已存在的模型文件: {model_path}")
            choice = input("是否重新训练模型？(y/n，默认n): ").strip().lower()
            if choice not in ['y', 'yes']:
                print("跳过训练，使用现有模型")
                self.steps_completed.append("train")
                return True
        
        self.wait_for_user("准备开始训练模型...")
        
        os.chdir(self.project_root / "train")
        
        print("正在训练模型...")
        try:
            # 使用实时输出，不缓冲
            process = subprocess.Popen([sys.executable, "train_model.py"], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True, 
                                     bufsize=1,
                                     universal_newlines=True)
            
            # 实时打印输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            return_code = process.poll()
            if return_code != 0:
                print(f"训练失败，退出码: {return_code}")
                return False
            
            # 检查模型文件是否生成
            model_path = self.project_root / "models" / "mnist_model.pth"
            if model_path.exists():
                print(f"✓ 模型已保存: {model_path}")
                self.steps_completed.append("train")
                return True
            else:
                print("✗ 模型文件未生成")
                return False
                
        except Exception as e:
            print(f"训练过程出错: {e}")
            return False
        finally:
            os.chdir(self.project_root)
    
    def step2_quantize_model(self):
        """步骤2: 模型量化"""
        self.print_step(2, "模型量化", 
                       "量化可以减少模型大小并提高推理速度。\n"
                       "我们将使用PyTorch的动态量化功能。")
        
        self.wait_for_user("准备开始量化...")
        
        os.chdir(self.project_root / "train")
        
        print("正在量化模型...")
        try:
            # 首先尝试标准量化
            process = subprocess.Popen([sys.executable, "quantize_model.py"], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True, 
                                     bufsize=1,
                                     universal_newlines=True)
            
            # 实时打印输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            return_code = process.poll()
            if return_code != 0:
                print(f"\n⚠️  量化失败，退出码: {return_code}")
                print("🔄 量化步骤失败，但可以继续使用原始模型进行后续步骤")
                
                # 检查原始模型是否存在
                original_model = self.project_root / "models" / "mnist_model.pth"
                if original_model.exists():
                    print("✓ 原始模型存在，可以继续后续步骤")
                    print("📝 注意: 将使用未量化的模型进行推理")
                    self.steps_completed.append("quantize")
                    return True
                else:
                    print("❌ 原始模型也不存在，请先完成训练步骤")
                    return False
            
            # 检查量化模型文件
            quantized_path = self.project_root / "models" / "mnist_quantized.pth"
            if quantized_path.exists():
                print(f"✓ 量化模型已保存: {quantized_path}")
                self.steps_completed.append("quantize")
                return True
            else:
                print("✗ 量化模型文件未生成")
                return False
                
        except Exception as e:
            print(f"量化过程出错: {e}")
            return False
        finally:
            os.chdir(self.project_root)
    
    def step3_export_onnx(self):
        """步骤3: 导出ONNX模型"""
        self.print_step(3, "导出ONNX模型", 
                       "ONNX是一个开放的神经网络交换格式，\n"
                       "它允许我们在不同的推理引擎之间使用同一个模型。")
        
        self.wait_for_user("准备导出ONNX模型...")
        
        os.chdir(self.project_root / "train")
        
        print("正在导出ONNX模型...")
        try:
            # 使用实时输出
            process = subprocess.Popen([sys.executable, "export_onnx.py"], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True, 
                                     bufsize=1,
                                     universal_newlines=True)
            
            # 实时打印输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            return_code = process.poll()
            if return_code != 0:
                print(f"ONNX导出失败，退出码: {return_code}")
                return False
            
            # 检查ONNX模型文件
            onnx_path = self.project_root / "models" / "mnist_model.onnx"
            if onnx_path.exists():
                print(f"✓ ONNX模型已保存: {onnx_path}")
                print(f"模型大小: {onnx_path.stat().st_size / 1024:.1f} KB")
                self.steps_completed.append("onnx")
                return True
            else:
                print("✗ ONNX模型文件未生成")
                return False
                
        except Exception as e:
            print(f"ONNX导出过程出错: {e}")
            return False
        finally:
            os.chdir(self.project_root)
    
    def step4_python_inference(self):
        """步骤4: Python推理测试"""
        self.print_step(4, "Python ONNX推理", 
                       "现在我们使用ONNX Runtime Python API来测试推理性能。\n"
                       "这将为我们提供准确率和性能基准。")
        
        self.wait_for_user("准备开始Python推理测试...")
        
        os.chdir(self.project_root / "inference")
        
        print("正在执行Python推理...")
        try:
            # 使用实时输出
            process = subprocess.Popen([sys.executable, "python_inference.py"], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True, 
                                     bufsize=1,
                                     universal_newlines=True)
            
            # 实时打印输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            return_code = process.poll()
            if return_code != 0:
                print(f"Python推理失败，退出码: {return_code}")
                return False
            
            # 检查结果文件
            results_path = self.project_root / "results" / "python_inference_results.json"
            if results_path.exists():
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    
                summary = results['summary']
                print(f"\n📊 Python推理结果:")
                print(f"准确率: {summary['accuracy']:.2%}")
                print(f"平均推理时间: {summary['average_inference_time_ms']:.2f} ms")
                print(f"推理速度: {summary['fps']:.1f} FPS")
                
                self.steps_completed.append("python_inference")
                return True
            else:
                print("✗ 推理结果文件未生成")
                return False
                
        except Exception as e:
            print(f"Python推理过程出错: {e}")
            return False
        finally:
            os.chdir(self.project_root)
    
    def step5_setup_cpp_environment(self):
        """步骤5: 设置统一版本编译环境"""
        self.print_step(5, "设置统一版本编译环境", 
                       "检查跨平台编译环境，支持 macOS 和 Android 统一版本编译。\n"
                       "统一版本架构使用单一源码支持多平台部署。")
        
        print("🔍 检查编译环境...")
        
        # 显示平台支持状态
        print(f"✓ macOS 编译: {self.platforms_available['macos']}")
        print(f"{'✓' if self.platforms_available['onnxruntime'] else '✗'} ONNX Runtime: {self.platforms_available['onnxruntime']}")
        print(f"{'✓' if self.platforms_available['android'] else '✗'} Android 编译: {self.platforms_available['android']}")
        
        # 检查必要的工具
        missing_tools = []
        if not self.platforms_available['onnxruntime']:
            missing_tools.append("ONNX Runtime")
        
        try:
            subprocess.run(["cmake", "--version"], capture_output=True, check=True)
            print("✓ CMake 已安装")
        except:
            missing_tools.append("CMake")
            print("✗ CMake 未安装")
        
        if missing_tools:
            print(f"\n⚠️  缺少工具: {', '.join(missing_tools)}")
            print("安装命令:")
            if "ONNX Runtime" in missing_tools:
                print("  brew install onnxruntime")
            if "CMake" in missing_tools:
                print("  brew install cmake")
            if not self.platforms_available['android']:
                print("  brew install --cask android-platform-tools")
                print("  brew install --cask android-ndk")
            
            choice = input("是否继续进行可用平台的编译？(y/n): ").strip().lower()
            return choice in ['y', 'yes']
        
        print("✅ 编译环境准备就绪")
        return True
    
    def step6_compile_cpp(self):
        """步骤6: 编译统一版本程序"""
        self.print_step(6, "编译统一版本推理程序", 
                       "使用统一版本编译脚本，支持 macOS 和 Android 双平台。\n"
                       "统一版本确保代码一致性和跨平台兼容性。")
        
        # 选择编译平台
        print("🎯 可用编译平台:")
        print("1. macOS (本地测试)")
        if self.platforms_available['android']:
            print("2. Android (移动部署)")
            print("3. 全部平台")
        
        choice = input("请选择编译平台 (默认1): ").strip()
        
        platforms_to_build = []
        if choice == "2" and self.platforms_available['android']:
            platforms_to_build = ["android"]
        elif choice == "3" and self.platforms_available['android']:
            platforms_to_build = ["macos", "android"]
        else:
            platforms_to_build = ["macos"]
        
        success_count = 0
        for platform in platforms_to_build:
            print(f"\n🔨 编译 {platform} 版本...")
            self.wait_for_user(f"准备编译 {platform} 版本...")
            
            try:
                # 使用统一版本编译脚本
                process = subprocess.Popen([
                    "./build.sh", platform
                ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                   text=True, bufsize=1, universal_newlines=True)
                
                # 实时打印输出
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output.strip())
                
                return_code = process.poll()
                if return_code == 0:
                    print(f"✅ {platform} 版本编译成功!")
                    success_count += 1
                    
                    # 检查生成的可执行文件
                    if platform == "macos":
                        cpp_exe = self.project_root / "inference" / "cpp_inference"
                        c_exe = self.project_root / "inference" / "c_inference"
                        if cpp_exe.exists() and c_exe.exists():
                            print(f"✓ macOS 可执行文件已生成")
                    elif platform == "android":
                        android_dir = self.project_root / "android_executables"
                        if android_dir.exists():
                            print(f"✓ Android 可执行文件已生成")
                else:
                    print(f"❌ {platform} 版本编译失败")
                    
            except Exception as e:
                print(f"❌ {platform} 编译过程出错: {e}")
        
        if success_count > 0:
            print(f"\n🎉 编译完成! 成功编译 {success_count}/{len(platforms_to_build)} 个平台")
            self.steps_completed.append("unified_compile")
            return True
        else:
            print("❌ 所有平台编译失败")
            return False
    
    def step7_test_inference(self):
        """步骤7: 跨平台推理测试"""
        self.print_step(7, "跨平台推理性能测试", 
                       "运行统一版本的推理程序，测试不同平台的性能表现。\n"
                       "支持 macOS 本地测试和 Android 设备测试。")
        
        print("🎯 可用测试平台:")
        print("1. macOS (本地)")
        if self.platforms_available['android']:
            print("2. Android (需要连接设备)")
            print("3. 完整跨平台测试")
        
        choice = input("请选择测试平台 (默认1): ").strip()
        
        if choice == "3" and self.platforms_available['android']:
            # 运行完整跨平台测试
            return self.run_cross_platform_test()
        elif choice == "2" and self.platforms_available['android']:
            # 单独运行Android测试
            return self.run_android_test()
        else:
            # 运行macOS测试
            return self.run_macos_test()
    
    def run_macos_test(self):
        """运行macOS本地测试"""
        print("🍎 运行 macOS 推理测试...")
        self.wait_for_user("准备测试 macOS 推理...")
        
        # 检查可执行文件
        cpp_exe = self.project_root / "inference" / "cpp_inference"
        c_exe = self.project_root / "inference" / "c_inference"
        
        if not cpp_exe.exists() or not c_exe.exists():
            print("❌ macOS 可执行文件不存在，请先编译")
            return False
        
        os.chdir(self.project_root / "inference")
        
        try:
            # 测试C++版本
            print("🔬 测试 C++ 版本...")
            result = subprocess.run(["./cpp_inference"], 
                                  capture_output=True, text=True, check=True)
            print("C++ 推理完成")
            
            # 测试C版本
            print("🔬 测试 C 版本...")
            result = subprocess.run(["./c_inference"], 
                                  capture_output=True, text=True, check=True)
            print("C 推理完成")
            
            # 显示结果文件
            self.show_inference_results("macos")
            self.steps_completed.append("macos_inference")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ macOS 推理失败: {e}")
            return False
        finally:
            os.chdir(self.project_root)
    
    def run_android_test(self):
        """运行Android设备测试"""
        print("📱 运行 Android 推理测试...")
        self.wait_for_user("准备测试 Android 推理...")
        
        try:
            # 使用部署脚本进行Android测试
            process = subprocess.Popen([
                "./deploy_and_test.sh", "android"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
               text=True, bufsize=1, universal_newlines=True)
            
            # 实时打印输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            return_code = process.poll()
            if return_code == 0:
                print("✅ Android 推理测试完成")
                self.show_inference_results("android")
                self.steps_completed.append("android_inference")
                return True
            else:
                print("❌ Android 推理测试失败")
                return False
                
        except Exception as e:
            print(f"❌ Android 测试过程出错: {e}")
            return False
    
    def run_cross_platform_test(self):
        """运行完整跨平台测试"""
        print("🌍 运行完整跨平台测试...")
        self.wait_for_user("准备运行完整跨平台测试...")
        
        try:
            # 使用完整测试脚本
            process = subprocess.Popen([
                "./run_all_platforms.sh"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
               text=True, bufsize=1, universal_newlines=True)
            
            # 实时打印输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            return_code = process.poll()
            if return_code == 0:
                print("✅ 跨平台测试完成")
                self.show_inference_results("all")
                self.steps_completed.append("cross_platform_inference")
                return True
            else:
                print("❌ 跨平台测试失败")
                return False
                
        except Exception as e:
            print(f"❌ 跨平台测试过程出错: {e}")
            return False
    
    def show_inference_results(self, platform_type):
        """显示推理结果"""
        results_dir = self.project_root / "results"
        
        print(f"\n📊 {platform_type.upper()} 推理结果:")
        print("=" * 50)
        
        # 根据平台类型显示对应结果
        result_files = []
        if platform_type == "macos":
            result_files = ["macos_cpp_results.txt", "macos_c_results.txt"]
        elif platform_type == "android":
            result_files = ["android_cpp_results.txt", "android_c_results.txt"]
        elif platform_type == "all":
            result_files = ["python_inference_results.json", 
                          "macos_cpp_results.txt", "macos_c_results.txt",
                          "android_cpp_results.txt", "android_c_results.txt"]
        
        for result_file in result_files:
            result_path = results_dir / result_file
            if result_path.exists():
                print(f"\n🔍 {result_file}:")
                if result_path.suffix == '.json':
                    with open(result_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        summary = data['summary']
                        print(f"  准确率: {summary['accuracy']:.2%}")
                        print(f"  平均时间: {summary['average_inference_time_ms']:.2f} ms")
                        print(f"  推理速度: {summary['fps']:.1f} FPS")
                else:
                    # 读取文本结果文件的关键信息
                    with open(result_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        import re
                        acc_match = re.search(r'准确率:\s*([0-9.]+)%', content)
                        time_match = re.search(r'平均推理时间:\s*([0-9.]+)\s*ms', content)
                        fps_match = re.search(r'推理速度:\s*([0-9.]+)\s*FPS', content)
                        
                        if acc_match:
                            print(f"  准确率: {acc_match.group(1)}%")
                        if time_match:
                            print(f"  平均时间: {time_match.group(1)} ms")
                        if fps_match:
                            print(f"  推理速度: {fps_match.group(1)} FPS")
            else:
                print(f"⚠️  {result_file} 不存在")
        
        # 检查是否有可视化图表
        chart_path = results_dir / "cross_platform_analysis.png"
        if chart_path.exists():
            print(f"\n📈 可视化图表已生成: {chart_path}")
            print("可使用以下命令查看:")
            print(f"  open {chart_path}")
        
        # 检查是否有分析报告
        report_path = results_dir / "unified_cross_platform_report.md"
        if report_path.exists():
            print(f"\n📋 详细分析报告: {report_path}")
     
    def step8_analyze_performance(self):
        """步骤8: 跨平台性能分析"""
        self.print_step(8, "跨平台性能深度分析", 
                       "生成详细的跨平台性能分析报告和可视化图表。\n"
                       "对比不同平台、不同语言的性能表现和优化建议。")
        
        self.wait_for_user("准备生成性能分析报告...")
        
        # 检查可用的结果文件
        results_dir = self.project_root / "results"
        available_results = []
        
        result_files = [
            ("python_inference_results.json", "Python"),
            ("macos_cpp_results.txt", "macOS C++"),
            ("macos_c_results.txt", "macOS C"),
            ("android_cpp_results.txt", "Android C++"),
            ("android_c_results.txt", "Android C")
        ]
        
        for file_name, description in result_files:
            if (results_dir / file_name).exists():
                available_results.append(description)
        
        print(f"📊 发现 {len(available_results)} 个测试结果:")
        for result in available_results:
            print(f"  ✓ {result}")
        
        if len(available_results) < 2:
            print("⚠️  需要至少2个平台的结果进行对比分析")
            print("请先运行推理测试获得更多结果")
            return False
        
        # 显示详细的性能对比
        print("\n📈 跨平台性能对比分析:")
        print("=" * 60)
        
        # 解析并显示各平台性能数据
        performance_data = {}
        
        for file_name, description in result_files:
            file_path = results_dir / file_name
            if file_path.exists():
                try:
                    if file_name.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            summary = data['summary']
                            performance_data[description] = {
                                'accuracy': summary['accuracy'],
                                'time_ms': summary['average_inference_time_ms'],
                                'fps': summary['fps']
                            }
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            import re
                            acc_match = re.search(r'准确率:\s*([0-9.]+)%', content)
                            time_match = re.search(r'平均推理时间:\s*([0-9.]+)\s*ms', content)
                            fps_match = re.search(r'推理速度:\s*([0-9.]+)\s*FPS', content)
                            
                            performance_data[description] = {
                                'accuracy': float(acc_match.group(1)) / 100 if acc_match else None,
                                'time_ms': float(time_match.group(1)) if time_match else None,
                                'fps': float(fps_match.group(1)) if fps_match else None
                            }
                except Exception as e:
                    print(f"⚠️  解析 {description} 结果时出错: {e}")
        
        # 显示性能表格
        print(f"{'平台':<15} {'准确率':<10} {'时间(ms)':<10} {'速度(FPS)':<12}")
        print("-" * 50)
        
        for platform, data in performance_data.items():
            acc_str = f"{data['accuracy']:.2%}" if data['accuracy'] else "N/A"
            time_str = f"{data['time_ms']:.2f}" if data['time_ms'] else "N/A"
            fps_str = f"{data['fps']:.1f}" if data['fps'] else "N/A"
            print(f"{platform:<15} {acc_str:<10} {time_str:<10} {fps_str:<12}")
        
        # 性能洞察分析
        print("\n🔍 性能洞察:")
        if len(performance_data) >= 2:
            # 找出最快和最慢的平台
            valid_times = {k: v['time_ms'] for k, v in performance_data.items() if v['time_ms']}
            if valid_times:
                fastest = min(valid_times, key=valid_times.get)
                slowest = max(valid_times, key=valid_times.get)
                speedup = valid_times[slowest] / valid_times[fastest]
                
                print(f"🏆 最快平台: {fastest} ({valid_times[fastest]:.2f}ms)")
                print(f"🐌 最慢平台: {slowest} ({valid_times[slowest]:.2f}ms)")
                print(f"📊 性能差异: {speedup:.2f}x")
                
                # 跨平台兼容性分析
                print(f"\n🌍 跨平台一致性:")
                accuracies = [v['accuracy'] for v in performance_data.values() if v['accuracy']]
                if len(accuracies) > 1:
                    max_acc = max(accuracies)
                    min_acc = min(accuracies)
                    print(f"准确率范围: {min_acc:.2%} - {max_acc:.2%}")
                    if max_acc - min_acc < 0.01:  # 1%以内
                        print("✅ 跨平台准确率高度一致")
                    else:
                        print("⚠️  跨平台准确率存在差异")
        
        # 检查是否生成了可视化图表
        chart_path = results_dir / "cross_platform_analysis.png"
        report_path = results_dir / "unified_cross_platform_report.md"
        
        if chart_path.exists():
            print(f"\n📊 可视化图表: {chart_path}")
            print("查看图表: open results/cross_platform_analysis.png")
        
        if report_path.exists():
            print(f"📋 详细报告: {report_path}")
        
        print("\n🎯 优化建议:")
        print("1. 移动端部署优先选择 C/C++ 版本")
        print("2. 开发阶段使用 Python 版本快速验证")
        print("3. 生产环境根据平台选择最优实现")
        print("4. 考虑模型量化进一步提升性能")
        
        self.steps_completed.append("performance_analysis")
        return True
    
    def run_tutorial(self):
        """运行完整教程"""
        print("🎓 MNIST模型部署教程 - 统一版本跨平台")
        print("=" * 50)
        print("本教程将引导您完成以下步骤:")
        print("1. 训练MNIST模型")
        print("2. 模型量化")
        print("3. 导出ONNX格式")
        print("4. Python推理测试")
        print("5. 设置统一版本编译环境")
        print("6. 编译统一版本程序 (macOS + Android)")
        print("7. 跨平台推理测试")
        print("8. 跨平台性能分析")
        
        self.wait_for_user("\n准备开始学习？")
        
        # 检查依赖
        if not self.check_dependencies():
            print("请安装缺失的依赖后重新运行")
            return False
        
        # 执行各个步骤
        steps = [
            self.step1_train_model,
            self.step2_quantize_model,
            self.step3_export_onnx,
            self.step4_python_inference,
            self.step5_setup_cpp_environment,
            self.step6_compile_cpp,
            self.step7_test_inference, # Corrected from step7_test_cpp_inference to step7_test_inference
            self.step8_analyze_performance # Corrected from step8_test_c_inference to step8_analyze_performance
        ]
        
        for i, step_func in enumerate(steps, 1):
            try:
                success = step_func()
                if not success:
                    print(f"\n❌ 步骤 {i} 失败")
                    retry = input("是否重试此步骤？(y/n): ").strip().lower()
                    if retry == 'y':
                        success = step_func()
                    
                    if not success:
                        print("跳过此步骤，继续下一步...")
                        continue
                
                print(f"✅ 步骤 {i} 完成")
                
            except KeyboardInterrupt:
                print("\n\n用户中断，教程结束")
                return False
            except Exception as e:
                print(f"\n❌ 步骤 {i} 出现意外错误: {e}")
                continue
        
        print("\n🎉 统一版本跨平台教程完成!")
        print(f"已完成步骤: {len(self.steps_completed)}/8")
        print("🌟 统一版本优势:")
        print("  ✅ 单一源码支持多平台")
        print("  ✅ 降低代码维护成本")
        print("  ✅ 保证跨平台一致性")
        print("  ✅ 便于新平台适配")
        print("\n🚀 接下来您可以:")
        print("1. 尝试在实际 Android 设备上部署")
        print("2. 探索其他深度学习模型")
        print("3. 优化模型性能和精度")
        print("4. 集成到生产应用中")
        print("\n感谢您完成 MNIST 统一版本跨平台部署教程！")
        
        return True

if __name__ == "__main__":
    tutorial = MNISTTutorial()
    tutorial.run_tutorial() 