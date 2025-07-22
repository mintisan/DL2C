#!/usr/bin/env python3
"""
MNIST模型部署教学脚本
一步一步引导用户完成从训练到部署的完整流程
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
                       "首先我们将训练一个简单的CNN模型来识别MNIST手写数字。\n"
                       "这个模型包含2个卷积层和2个全连接层。\n\n"
                       "选择训练模式:\n"
                       "1. 快速演示版 (1分钟，1000样本，1个epoch) - 推荐学习\n"
                       "2. 完整训练版 (5-10分钟，60000样本，5个epoch) - 更好精度")
        
        choice = input("请选择训练模式 (1/2，默认1): ").strip()
        if choice == "2":
            script_name = "train_model.py"
            print("选择完整训练版...")
        else:
            script_name = "train_model_quick.py"
            print("选择快速演示版...")
        
        self.wait_for_user("准备开始训练模型...")
        
        os.chdir(self.project_root / "train")
        
        print("正在训练模型...")
        try:
            # 使用实时输出，不缓冲
            process = subprocess.Popen([sys.executable, script_name], 
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
                print(f"\n⚠️  标准量化失败，使用兼容版本...")
                print("🔄 在macOS上使用模拟量化方法...")
                
                # 使用兼容的量化版本
                process = subprocess.Popen([sys.executable, "quantize_model_simple.py"], 
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
                    print(f"兼容量化也失败，退出码: {return_code}")
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
        """步骤5: 设置C++编译环境"""
        self.print_step(5, "设置C++编译环境", 
                       "为了编译C++版本，我们需要安装ONNX Runtime C++库。\n"
                       "这将允许我们创建更高性能的推理引擎。")
        
        print("检查ONNX Runtime C++库...")
        
        # 检查是否已经有ONNX Runtime
        homebrew_path = "/opt/homebrew/include/onnxruntime/onnxruntime_cxx_api.h"
        if os.path.exists(homebrew_path):
            print("✓ 发现Homebrew安装的ONNX Runtime")
            return True
        
        # 提示用户安装
        print("未找到ONNX Runtime C++库")
        print("\n安装选项:")
        print("1. 使用Homebrew安装 (推荐)")
        print("2. 手动下载预编译版本")
        
        choice = input("请选择安装方式 (1/2): ").strip()
        
        if choice == "1":
            print("正在使用Homebrew安装ONNX Runtime...")
            try:
                result = subprocess.run(["brew", "install", "onnxruntime"], 
                                      capture_output=True, text=True, check=True)
                print("✓ ONNX Runtime安装成功")
                return True
            except subprocess.CalledProcessError as e:
                print(f"Homebrew安装失败: {e}")
                print("请尝试手动安装或检查Homebrew配置")
                return False
        elif choice == "2":
            print("\n手动安装步骤:")
            print("1. 访问: https://github.com/microsoft/onnxruntime/releases")
            print("2. 下载适合您平台的预编译版本")
            print("3. 解压到build目录下")
            print("例如: build/onnxruntime-osx-arm64-1.16.0/")
            
            self.wait_for_user("完成手动安装后按回车继续...")
            return True
        else:
            print("无效的选择")
            return False
    
    def step6_compile_cpp(self):
        """步骤6: 编译C++版本"""
        self.print_step(6, "编译C++推理程序", 
                       "现在我们将编译C++版本的推理程序。\n"
                       "C++版本通常比Python版本更快，适合生产环境。")
        
        self.wait_for_user("准备编译C++程序...")
        
        # 创建build目录
        build_dir = self.project_root / "build" / "build_macos"
        build_dir.mkdir(parents=True, exist_ok=True)
        
        os.chdir(build_dir)
        
        print("正在配置CMake...")
        try:
            # 配置
            result = subprocess.run([
                "cmake", 
                "-DCMAKE_BUILD_TYPE=Release",
                ".."
            ], capture_output=True, text=True, check=True)
            
            print("CMake配置成功")
            
            # 编译
            print("正在编译...")
            result = subprocess.run([
                "make", "-j4"
            ], capture_output=True, text=True, check=True)
            
            print("编译成功!")
            
            # 检查可执行文件
            exe_path = build_dir / "bin" / "mnist_inference_cpp"
            if exe_path.exists():
                print(f"✓ 可执行文件生成: {exe_path}")
                self.steps_completed.append("cpp_compile")
                return True
            else:
                print("✗ 可执行文件未生成")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"编译失败: {e}")
            print(f"错误输出: {e.stderr}")
            print("\n可能的解决方案:")
            print("1. 确保已安装ONNX Runtime")
            print("2. 检查CMake和编译器是否正确安装")
            print("3. 查看详细错误信息")
            return False
        finally:
            os.chdir(self.project_root)
    
    def step7_test_cpp_inference(self):
        """步骤7: 测试C++推理"""
        self.print_step(7, "测试C++推理性能", 
                       "运行C++推理程序并比较与Python版本的性能差异。")
        
        self.wait_for_user("准备测试C++推理...")
        
        exe_path = self.project_root / "build" / "build_macos" / "bin" / "mnist_inference_cpp"
        
        if not exe_path.exists():
            print(f"✗ 可执行文件不存在: {exe_path}")
            return False
        
        print("正在运行C++推理测试...")
        try:
            result = subprocess.run([str(exe_path)], 
                                  capture_output=True, text=True, check=True, 
                                  cwd=str(self.project_root))
            print("C++推理输出:")
            print(result.stdout)
            
            # 检查结果文件
            results_path = self.project_root / "results" / "cpp_inference_results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    
                summary = results['summary']
                print(f"\n📊 C++推理结果:")
                print(f"平均推理时间: {summary['average_inference_time_ms']:.2f} ms")
                print(f"推理速度: {summary['fps']:.1f} FPS")
                
                self.steps_completed.append("cpp_inference")
                return True
            else:
                print("✓ C++推理完成（结果文件生成可选）")
                self.steps_completed.append("cpp_inference")
                return True
                
        except subprocess.CalledProcessError as e:
            print(f"C++推理失败: {e}")
            print(f"错误输出: {e.stderr}")
            return False
    
    def step8_test_c_inference(self):
        """步骤8: 测试C推理"""
        self.print_step(8, "测试C语言推理性能", 
                       "运行纯C语言推理程序，体验最底层的ONNX Runtime C API。\n"
                       "C语言版本通常有最好的跨平台兼容性。")
        
        self.wait_for_user("准备测试C语言推理...")
        
        exe_path = self.project_root / "build" / "build_macos" / "bin" / "mnist_inference_c"
        
        if not exe_path.exists():
            print(f"✗ 可执行文件不存在: {exe_path}")
            return False
        
        print("正在运行C语言推理测试...")
        try:
            result = subprocess.run([str(exe_path)], 
                                  capture_output=True, text=True, check=True, 
                                  cwd=str(self.project_root))
            print("C语言推理输出:")
            print(result.stdout)
            
            # 检查结果文件
            results_path = self.project_root / "results" / "c_inference_results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    
                summary = results['summary']
                print(f"\n📊 C语言推理结果:")
                print(f"平均推理时间: {summary['average_inference_time_ms']:.2f} ms")
                print(f"推理速度: {summary['fps']:.1f} FPS")
                
                self.steps_completed.append("c_inference")
                return True
            else:
                print("✓ C语言推理完成（结果文件生成可选）")
                self.steps_completed.append("c_inference")
                return True
                
        except subprocess.CalledProcessError as e:
            print(f"C语言推理失败: {e}")
            print(f"错误输出: {e.stderr}")
            return False
    
    def step9_compare_results(self):
        """步骤9: 三种语言性能对比"""
        self.print_step(9, "三种语言推理性能对比", 
                       "比较Python、C++和C语言三种实现的推理性能差异。")
        
        # 加载三种语言的结果
        python_results_path = self.project_root / "results" / "python_inference_results.json"
        cpp_results_path = self.project_root / "results" / "cpp_inference_results.json"
        c_results_path = self.project_root / "results" / "c_inference_results.json"
        
        python_data = None
        cpp_data = None
        c_data = None
        
        if python_results_path.exists():
            with open(python_results_path, 'r', encoding='utf-8') as f:
                python_data = json.load(f)
        
        if cpp_results_path.exists():
            with open(cpp_results_path, 'r') as f:
                cpp_data = json.load(f)
                
        if c_results_path.exists():
            with open(c_results_path, 'r') as f:
                c_data = json.load(f)
        
        print("\n📊 三种语言性能对比报告:")
        print("=" * 60)
        
        if python_data:
            py_summary = python_data['summary']
            print(f"🐍 Python (ONNX Runtime Python API):")
            print(f"    准确率: {py_summary['accuracy']:.2%}")
            print(f"    平均时间: {py_summary['average_inference_time_ms']:.2f} ms")
            print(f"    推理速度: {py_summary['fps']:.1f} FPS")
        
        if cpp_data:
            cpp_summary = cpp_data['summary']
            print(f"⚡ C++ (ONNX Runtime C++ API):")
            print(f"    平均时间: {cpp_summary['average_inference_time_ms']:.2f} ms")
            print(f"    推理速度: {cpp_summary['fps']:.1f} FPS")
        
        if c_data:
            c_summary = c_data['summary']
            print(f"🔧 C (ONNX Runtime C API):")
            print(f"    平均时间: {c_summary['average_inference_time_ms']:.2f} ms")
            print(f"    推理速度: {c_summary['fps']:.1f} FPS")
        
        # 性能对比分析
        if python_data and cpp_data and c_data:
            py_time = python_data['summary']['average_inference_time_ms']
            cpp_time = cpp_data['summary']['average_inference_time_ms']
            c_time = c_data['summary']['average_inference_time_ms']
            
            print(f"\n📈 性能提升对比 (相对于Python):")
            print(f"    C++加速: {py_time / cpp_time:.2f}x")
            print(f"    C语言加速: {py_time / c_time:.2f}x")
            
            if cpp_time != 0 and c_time != 0:
                print(f"    C vs C++: {cpp_time / c_time:.2f}x")
        
        print("\n🎉 完整推理性能测试完成!")
        print("\n🚀 接下来可以尝试:")
        print("1. Android NDK编译 (需要Android开发环境)")
        print("2. 模型优化和量化技术")
        print("3. 集成到实际移动应用中")
        print("4. 尝试其他深度学习模型")
        
        return True
    
    def run_tutorial(self):
        """运行完整教程"""
        print("🎓 MNIST模型部署教程")
        print("=" * 50)
        print("本教程将引导您完成以下步骤:")
        print("1. 训练MNIST模型")
        print("2. 模型量化")
        print("3. 导出ONNX格式")
        print("4. Python推理测试")
        print("5. 设置C++环境")
        print("6. 编译C++/C程序")
        print("7. C++推理测试")
        print("8. C语言推理测试")
        print("9. 三种语言性能对比")
        
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
            self.step7_test_cpp_inference,
            self.step8_test_c_inference,
            self.step9_compare_results
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
        
        print("\n🎉 教程完成!")
        print(f"已完成步骤: {len(self.steps_completed)}/9")
        print("感谢您完成MNIST模型部署教程！")
        
        return True

if __name__ == "__main__":
    tutorial = MNISTTutorial()
    tutorial.run_tutorial() 