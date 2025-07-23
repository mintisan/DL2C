#!/bin/bash
# 统一版本 ONNX Runtime 编译脚本
# 支持 macOS 本地编译和 Android 交叉编译

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== 统一版本 ONNX Runtime 编译脚本 ===${NC}"

# 设置变量
PROJECT_DIR=$(pwd)
BUILD_DIR=$(pwd)
ANDROID_ABI="arm64-v8a"
ANDROID_API=21
BUILD_TYPE="Release"

# 默认构建Android版本，除非指定macos
BUILD_TARGET="android"
if [[ "$1" == "macos" ]]; then
    BUILD_TARGET="macos"
fi

echo -e "${YELLOW}构建目标: $BUILD_TARGET${NC}"

# 检查环境
check_environment() {
    echo -e "${YELLOW}检查编译环境...${NC}"
    
    if [[ "$BUILD_TARGET" == "android" ]]; then
        # 检查Android环境
        ANDROID_NDK_HOME=${ANDROID_NDK_HOME:-"/opt/homebrew/share/android-ndk"}
        
        if [ ! -d "$ANDROID_NDK_HOME" ]; then
            echo -e "${RED}错误: Android NDK 未找到在 $ANDROID_NDK_HOME${NC}"
            echo "请设置正确的 ANDROID_NDK_HOME 环境变量"
            exit 1
        fi
        
        echo -e "${GREEN}✓ Android NDK: $ANDROID_NDK_HOME${NC}"
        
        # 检查 ONNX Runtime Android 版本
        ONNXRUNTIME_BUILD_DIR="$HOME/Workplaces/onnxruntime/build/Android/Release"
        if [ ! -f "$ONNXRUNTIME_BUILD_DIR/libonnxruntime_session.a" ]; then
            echo -e "${RED}错误: ONNX Runtime Android 版本未找到${NC}"
            echo "请先成功编译 ONNX Runtime Android 版本"
            exit 1
        fi
        
        echo -e "${GREEN}✓ ONNX Runtime Android 库存在${NC}"
        
        # 检查 adb 连接（可选）
        DEVICE_COUNT=$(adb devices 2>/dev/null | grep -c "device$" || echo "0")
        if [ "$DEVICE_COUNT" -eq 0 ] 2>/dev/null; then
            echo -e "${YELLOW}警告: 没有连接的 Android 设备${NC}"
            echo "编译将继续，但无法直接部署"
        else
            echo -e "${GREEN}✓ Android 设备已连接${NC}"
        fi
        
    else
        # 检查macOS环境
        echo -e "${BLUE}检查macOS环境...${NC}"
        
        # 检查ONNX Runtime是否安装
        if ! pkg-config --exists onnxruntime 2>/dev/null; then
            # 尝试查找系统安装的ONNX Runtime
            ONNX_DIRS=("/usr/local" "/opt/homebrew" "/usr")
            FOUND_ONNX=false
            
            for dir in "${ONNX_DIRS[@]}"; do
                if [ -f "$dir/lib/libonnxruntime.dylib" ] || [ -f "$dir/lib/libonnxruntime.so" ]; then
                    echo -e "${GREEN}✓ 找到ONNX Runtime: $dir${NC}"
                    FOUND_ONNX=true
                    break
                fi
            done
            
            if [ "$FOUND_ONNX" = false ]; then
                echo -e "${YELLOW}警告: 系统未安装ONNX Runtime${NC}"
                echo "请安装ONNX Runtime或确保在标准路径中可用"
                echo "macOS安装命令: brew install onnxruntime"
            fi
        else
            echo -e "${GREEN}✓ ONNX Runtime pkg-config配置存在${NC}"
        fi
        
        # 检查编译工具
        if ! command -v cmake &> /dev/null; then
            echo -e "${RED}错误: cmake 未安装${NC}"
            exit 1
        fi
        
        if ! command -v make &> /dev/null; then
            echo -e "${RED}错误: make 未安装${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}✓ macOS 编译环境检查通过${NC}"
    fi
}

# 创建输出目录
create_directories() {
    echo -e "${YELLOW}创建输出目录...${NC}"
    
    if [[ "$BUILD_TARGET" == "android" ]]; then
        mkdir -p "${PROJECT_DIR}/android_executables/${ANDROID_ABI}"
    else
        mkdir -p "${PROJECT_DIR}/inference"
    fi
    
    mkdir -p "${PROJECT_DIR}/results"
    
    echo -e "${GREEN}✓ 目录创建完成${NC}"
}

# 编译统一版本
build_project() {
    echo -e "${YELLOW}编译统一版本 ($BUILD_TARGET)...${NC}"
    
    # 清理之前的构建
    BUILD_DIR_NAME="build/build_${BUILD_TARGET}"
    rm -rf "$BUILD_DIR_NAME"
    mkdir -p "$BUILD_DIR_NAME"
    cd "$BUILD_DIR_NAME"
    
    # 复制 CMakeLists.txt 到构建目录
    cp ../CMakeLists.txt ./CMakeLists.txt
    
    # 运行 CMake 配置
    echo -e "${BLUE}配置 CMake ($BUILD_TARGET)...${NC}"
    
    if [[ "$BUILD_TARGET" == "android" ]]; then
        # Android 交叉编译配置
        cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
              -DANDROID_ABI=$ANDROID_ABI \
              -DANDROID_NATIVE_API_LEVEL=$ANDROID_API \
              -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
              -DCMAKE_CXX_STANDARD=17 \
              -DANDROID=ON \
              .
    else
        # macOS 本地编译配置
        cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
              -DCMAKE_CXX_STANDARD=17 \
              .
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: CMake 配置失败${NC}"
        exit 1
    fi
    
    # 编译
    echo -e "${BLUE}开始编译...${NC}"
    if command -v nproc &> /dev/null; then
        make -j$(nproc)
    else
        # macOS 使用 sysctl
        make -j$(sysctl -n hw.ncpu)
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 编译失败${NC}"
        exit 1
    fi
    
    cd ..
    
    echo -e "${GREEN}✓ 统一版本 ($BUILD_TARGET) 编译成功${NC}"
}

# 检查可执行文件生成
check_executables() {
    echo -e "${YELLOW}检查可执行文件生成...${NC}"
    
    if [[ "$BUILD_TARGET" == "android" ]]; then
        # Android 版本
        TARGET_DIR="${PROJECT_DIR}/android_executables/${ANDROID_ABI}"
        
        # 检查C++版本
        if [ -f "$TARGET_DIR/cpp_inference" ]; then
            chmod +x "$TARGET_DIR/cpp_inference"
            echo -e "${GREEN}✓ Android C++ 推理程序生成成功${NC}"
        else
            echo -e "${RED}错误: Android C++ 推理可执行文件未生成${NC}"
        fi
        
        # 检查C版本
        if [ -f "$TARGET_DIR/c_inference" ]; then
            chmod +x "$TARGET_DIR/c_inference"
            echo -e "${GREEN}✓ Android C 推理程序生成成功${NC}"
        else
            echo -e "${RED}错误: Android C 推理可执行文件未生成${NC}"
        fi
        
    else
        # macOS 版本
        TARGET_DIR="${PROJECT_DIR}/inference"
        
        # 检查C++版本
        if [ -f "$TARGET_DIR/cpp_inference" ]; then
            chmod +x "$TARGET_DIR/cpp_inference"
            echo -e "${GREEN}✓ macOS C++ 推理程序生成成功${NC}"
        else
            echo -e "${RED}错误: macOS C++ 推理可执行文件未生成${NC}"
        fi
        
        # 检查C版本
        if [ -f "$TARGET_DIR/c_inference" ]; then
            chmod +x "$TARGET_DIR/c_inference"
            echo -e "${GREEN}✓ macOS C 推理程序生成成功${NC}"
        else
            echo -e "${RED}错误: macOS C 推理可执行文件未生成${NC}"
        fi
    fi
}

# 显示结果
show_results() {
    echo -e "\n${GREEN}=== 编译完成！===${NC}"
    
    if [[ "$BUILD_TARGET" == "android" ]]; then
        TARGET_DIR="${PROJECT_DIR}/android_executables/${ANDROID_ABI}"
        echo -e "${BLUE}Android 生成的文件:${NC}"
        ls -la "$TARGET_DIR/" | grep "_inference" || echo "未找到推理可执行文件"
        
        echo -e "\n${BLUE}文件大小:${NC}"
        ls -lh "$TARGET_DIR/cpp_inference" 2>/dev/null || echo "C++ 推理文件不存在"
        ls -lh "$TARGET_DIR/c_inference" 2>/dev/null || echo "C 推理文件不存在"
        
        echo -e "\n${BLUE}架构信息:${NC}"
        echo "C++ 推理程序:"
        file "$TARGET_DIR/cpp_inference" 2>/dev/null || echo "无法获取架构信息"
        echo "C 推理程序:"
        file "$TARGET_DIR/c_inference" 2>/dev/null || echo "无法获取架构信息"
        
        echo -e "\n${GREEN}下一步: 运行 ./deploy_and_test.sh 部署到 Android 设备${NC}"
        
    else
        TARGET_DIR="${PROJECT_DIR}/inference"
        echo -e "${BLUE}macOS 生成的文件:${NC}"
        ls -la "$TARGET_DIR/" | grep "_inference" || echo "未找到推理可执行文件"
        
        echo -e "\n${BLUE}文件大小:${NC}"
        ls -lh "$TARGET_DIR/cpp_inference" 2>/dev/null || echo "C++ 推理文件不存在"
        ls -lh "$TARGET_DIR/c_inference" 2>/dev/null || echo "C 推理文件不存在"
        
        echo -e "\n${GREEN}下一步: 直接运行推理测试${NC}"
        echo "  C++ 版本: cd inference && ./cpp_inference"
        echo "  C 版本:   cd inference && ./c_inference"
    fi
}

# 主函数
main() {
    echo "开始时间: $(date)"
    echo "构建目标: $BUILD_TARGET"
    
    check_environment
    create_directories
    build_project
    check_executables
    show_results
    
    echo "结束时间: $(date)"
    echo -e "${GREEN}统一版本 ($BUILD_TARGET) 编译脚本执行完成！${NC}"
}

# 显示使用说明
show_usage() {
    echo "使用方法:"
    echo "  $0 [macos|android]"
    echo ""
    echo "参数:"
    echo "  macos    - 编译macOS本地版本 (默认)"
    echo "  android  - 编译Android交叉编译版本"
    echo ""
    echo "示例:"
    echo "  $0 macos     # 编译macOS版本"
    echo "  $0 android   # 编译Android版本"
    echo "  $0           # 默认编译Android版本"
}

# 处理命令行参数
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

if [[ -n "$1" ]] && [[ "$1" != "macos" ]] && [[ "$1" != "android" ]]; then
    echo -e "${RED}错误: 无效的构建目标 '$1'${NC}"
    show_usage
    exit 1
fi

# 执行主函数
main "$@" 