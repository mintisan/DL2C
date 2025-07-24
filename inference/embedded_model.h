/*
 * 嵌入式ONNX模型数据头文件
 * 生成时间: 2025-07-24 14:42:27
 */

#ifndef EMBEDDED_MODEL_DATA_H
#define EMBEDDED_MODEL_DATA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// 嵌入式模型数据声明
extern const unsigned char mnist_model_data[];
extern const size_t mnist_model_data_size;

// 获取嵌入式模型数据的函数
const unsigned char* get_embedded_model_data(void);
size_t get_embedded_model_size(void);

#ifdef __cplusplus
}
#endif

#endif // EMBEDDED_MODEL_DATA_H
