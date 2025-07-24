#!/usr/bin/env python3
"""
ONNX模型转C数组工具
将ONNX模型文件转换为C语言数组，用于嵌入到静态库中
"""

import sys
import os
from pathlib import Path

def onnx_to_c_array(onnx_file_path, output_c_file, array_name="embedded_model_data"):
    """
    将ONNX模型文件转换为C数组
    
    Args:
        onnx_file_path: ONNX模型文件路径
        output_c_file: 输出的C文件路径
        array_name: C数组名称
    """
    
    # 读取ONNX文件
    try:
        with open(onnx_file_path, 'rb') as f:
            model_data = f.read()
    except FileNotFoundError:
        print(f"❌ 错误: 找不到模型文件 {onnx_file_path}")
        return False
    except Exception as e:
        print(f"❌ 错误: 读取模型文件失败 - {e}")
        return False
    
    model_size = len(model_data)
    print(f"📄 模型文件: {onnx_file_path}")
    print(f"📏 模型大小: {model_size:,} bytes ({model_size/1024/1024:.1f} MB)")
    
    # 生成C文件内容
    c_content = f"""/*
 * 自动生成的嵌入式ONNX模型数据
 * 原始文件: {os.path.basename(onnx_file_path)}
 * 文件大小: {model_size:,} bytes ({model_size/1024/1024:.1f} MB)
 * 生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 * 
 * 注意: 此文件由工具自动生成，请勿手动编辑
 */

#include <stddef.h>

// 嵌入式模型数据
const unsigned char {array_name}[] = {{
"""
    
    # 将二进制数据转换为C数组格式
    bytes_per_line = 16
    for i in range(0, model_size, bytes_per_line):
        chunk = model_data[i:i + bytes_per_line]
        hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
        c_content += f"    {hex_values}"
        
        if i + bytes_per_line < model_size:
            c_content += ","
        
        c_content += f"  // bytes {i}-{min(i + bytes_per_line - 1, model_size - 1)}\n"
    
    c_content += f"""
}};

// 模型数据大小
const size_t {array_name}_size = {model_size};

// 获取嵌入式模型数据的函数
const unsigned char* get_embedded_model_data(void) {{
    return {array_name};
}}

// 获取嵌入式模型数据大小的函数
size_t get_embedded_model_size(void) {{
    return {array_name}_size;
}}
"""
    
    # 写入C文件
    try:
        output_dir = os.path.dirname(output_c_file)
        if output_dir:  # 只有在有目录路径时才创建目录
            os.makedirs(output_dir, exist_ok=True)
        with open(output_c_file, 'w') as f:
            f.write(c_content)
        
        print(f"✅ C数组文件已生成: {output_c_file}")
        print(f"📊 数组名称: {array_name}")
        print(f"📊 数组大小: {array_name}_size = {model_size}")
        return True
        
    except Exception as e:
        print(f"❌ 错误: 写入C文件失败 - {e}")
        return False

def generate_header_file(output_h_file, array_name="embedded_model_data"):
    """生成对应的头文件"""
    
    header_content = f"""/*
 * 嵌入式ONNX模型数据头文件
 * 生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 */

#ifndef EMBEDDED_MODEL_DATA_H
#define EMBEDDED_MODEL_DATA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {{
#endif

// 嵌入式模型数据声明
extern const unsigned char {array_name}[];
extern const size_t {array_name}_size;

// 获取嵌入式模型数据的函数
const unsigned char* get_embedded_model_data(void);
size_t get_embedded_model_size(void);

#ifdef __cplusplus
}}
#endif

#endif // EMBEDDED_MODEL_DATA_H
"""
    
    try:
        with open(output_h_file, 'w') as f:
            f.write(header_content)
        print(f"✅ 头文件已生成: {output_h_file}")
        return True
    except Exception as e:
        print(f"❌ 错误: 写入头文件失败 - {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("使用方法: python3 onnx_to_c_array.py <onnx_file> <output_c_file> [array_name]")
        print("示例: python3 onnx_to_c_array.py ../models/mnist_model.onnx embedded_model.c")
        sys.exit(1)
    
    onnx_file = sys.argv[1]
    output_c_file = sys.argv[2]
    array_name = sys.argv[3] if len(sys.argv) > 3 else "embedded_model_data"
    
    # 生成头文件路径
    output_h_file = output_c_file.replace('.c', '.h')
    
    print("🔄 === ONNX模型转C数组工具 ===")
    print()
    
    # 转换模型为C数组
    if onnx_to_c_array(onnx_file, output_c_file, array_name):
        # 生成头文件
        generate_header_file(output_h_file, array_name)
        print()
        print("🎊 === 转换成功！===")
        print(f"📁 生成的文件:")
        print(f"   📄 {output_c_file}")
        print(f"   📄 {output_h_file}")
        print()
        print("💡 使用说明:")
        print(f"   1. 在你的C代码中 #include \"{os.path.basename(output_h_file)}\"")
        print(f"   2. 调用 get_embedded_model_data() 获取模型数据指针")
        print(f"   3. 调用 get_embedded_model_size() 获取模型数据大小")
        print(f"   4. 使用 OrtCreateSessionFromArray() 而不是 OrtCreateSession()")
    else:
        print("❌ 转换失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 