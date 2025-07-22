#!/bin/bash
# Android 真实 ONNX Runtime 交叉编译脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Android 真实 ONNX Runtime 交叉编译脚本 ===${NC}"

# 设置变量
ANDROID_NDK_HOME=${ANDROID_NDK_HOME:-"/opt/homebrew/share/android-ndk"}
PROJECT_DIR=$(pwd)/..
BUILD_DIR=$(pwd)
ANDROID_ABI="arm64-v8a"
ANDROID_API=21

# 检查环境
check_environment() {
    echo -e "${YELLOW}检查编译环境...${NC}"
    
    if [ ! -d "$ANDROID_NDK_HOME" ]; then
        echo -e "${RED}错误: Android NDK 未找到在 $ANDROID_NDK_HOME${NC}"
        echo "请设置正确的 ANDROID_NDK_HOME 环境变量"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Android NDK: $ANDROID_NDK_HOME${NC}"
    
    # 检查 ONNX Runtime 是否构建成功
    ONNXRUNTIME_BUILD_DIR="/Users/mintisan/Workplaces/onnxruntime/build/Android/Release"
    if [ ! -f "$ONNXRUNTIME_BUILD_DIR/libonnxruntime_session.a" ]; then
        echo -e "${RED}错误: ONNX Runtime Android 版本未找到${NC}"
        echo "请先成功编译 ONNX Runtime Android 版本"
        exit 1
    fi
    
    echo -e "${GREEN}✓ ONNX Runtime Android 库存在${NC}"
    
    # 检查 adb 连接
    DEVICE_COUNT=$(adb devices | grep -c "device$" || echo "0")
    if [ "$DEVICE_COUNT" -eq 0 ]; then
        echo -e "${YELLOW}警告: 没有连接的 Android 设备${NC}"
        echo "编译将继续，但无法直接部署"
    else
        echo -e "${GREEN}✓ Android 设备已连接${NC}"
    fi
}

# 创建输出目录
create_directories() {
    echo -e "${YELLOW}创建输出目录...${NC}"
    
    mkdir -p "${PROJECT_DIR}/android_executables/${ANDROID_ABI}"
    mkdir -p "${PROJECT_DIR}/results"
    
    echo -e "${GREEN}✓ 目录创建完成${NC}"
}

# 编译真实 ONNX Runtime 版本
build_real_onnx() {
    echo -e "${YELLOW}编译 Android 真实 ONNX Runtime 版本...${NC}"
    
    # 清理之前的构建
    rm -rf build_android_real
    mkdir -p build_android_real
    cd build_android_real
    
    # 复制 CMakeLists.txt 到构建目录
    cp ../CMakeLists_android_real.txt ./CMakeLists.txt
    
    # 运行 CMake 配置
    echo -e "${BLUE}配置 CMake...${NC}"
    cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
          -DANDROID_ABI=$ANDROID_ABI \
          -DANDROID_NATIVE_API_LEVEL=$ANDROID_API \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_STANDARD=17 \
          .
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: CMake 配置失败${NC}"
        exit 1
    fi
    
    # 编译
    echo -e "${BLUE}开始编译...${NC}"
    make -j$(nproc)
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 编译失败${NC}"
        exit 1
    fi
    
    cd ..
    
    echo -e "${GREEN}✓ Android 真实 ONNX Runtime 编译成功${NC}"
}

# 复制可执行文件
copy_executables() {
    echo -e "${YELLOW}复制可执行文件...${NC}"
    
    if [ -f "build_android_real/android_real_onnx_inference" ]; then
        cp build_android_real/android_real_onnx_inference "${PROJECT_DIR}/android_executables/${ANDROID_ABI}/"
        chmod +x "${PROJECT_DIR}/android_executables/${ANDROID_ABI}/android_real_onnx_inference"
        echo -e "${GREEN}✓ android_real_onnx_inference 已复制${NC}"
    else
        echo -e "${RED}错误: 可执行文件未生成${NC}"
        exit 1
    fi
}

# 显示结果
show_results() {
    echo -e "\n${GREEN}=== 编译完成！===${NC}"
    echo -e "${BLUE}生成的文件:${NC}"
    ls -la "${PROJECT_DIR}/android_executables/${ANDROID_ABI}/"
    
    echo -e "\n${BLUE}文件大小:${NC}"
    ls -lh "${PROJECT_DIR}/android_executables/${ANDROID_ABI}/android_real_onnx_inference" 2>/dev/null || echo "文件不存在"
    
    echo -e "\n${BLUE}架构信息:${NC}"
    file "${PROJECT_DIR}/android_executables/${ANDROID_ABI}/android_real_onnx_inference" 2>/dev/null || echo "无法获取架构信息"
    
    echo -e "\n${GREEN}下一步: 运行 ./deploy_and_test_real_onnx.sh 部署到 Android 设备${NC}"
}

# 主函数
main() {
    echo "开始时间: $(date)"
    
    check_environment
    create_directories
    build_real_onnx
    copy_executables
    show_results
    
    echo "结束时间: $(date)"
    echo -e "${GREEN}Android 真实 ONNX Runtime 编译脚本执行完成！${NC}"
}

# 执行主函数
main "$@" 