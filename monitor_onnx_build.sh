#!/bin/bash
# ONNX Runtime Android编译监控脚本

ONNX_DIR="/Users/mintisan/Workplaces/onnxruntime"
LOG_FILE="$ONNX_DIR/build.log"
BUILD_PID=$(pgrep -f "build.sh.*android" | head -1)

echo "=== ONNX Runtime Android 编译监控 ==="
echo "编译目录: $ONNX_DIR"
echo "日志文件: $LOG_FILE"
echo "编译进程PID: $BUILD_PID"

if [ -n "$BUILD_PID" ]; then
    echo "✅ 编译进程正在运行"
    
    # 显示最新的编译日志
    echo -e "\n📄 最新编译日志:"
    if [ -f "$LOG_FILE" ]; then
        tail -10 "$LOG_FILE"
    else
        echo "日志文件暂未生成"
    fi
    
    # 检查build目录
    echo -e "\n📁 Build目录状态:"
    if [ -d "$ONNX_DIR/build" ]; then
        find "$ONNX_DIR/build" -name "*.a" 2>/dev/null | wc -l | xargs echo "静态库文件数:"
        du -sh "$ONNX_DIR/build" 2>/dev/null | cut -f1 | xargs echo "目录大小:"
    else
        echo "Build目录尚未创建"
    fi
    
    # 检查关键库文件
    echo -e "\n🎯 关键文件检查:"
    CRITICAL_FILE="$ONNX_DIR/build/Android/Release/libonnxruntime_session.a"
    if [ -f "$CRITICAL_FILE" ]; then
        ls -lh "$CRITICAL_FILE" | awk '{print "libonnxruntime_session.a: " $5}'
        echo "🎉 关键库文件已生成！编译即将完成"
    else
        echo "⏳ 关键库文件尚未生成，编译仍在进行中..."
    fi
    
else
    echo "❌ 编译进程未找到"
    
    # 检查是否编译已完成
    if [ -f "$ONNX_DIR/build/Android/Release/libonnxruntime_session.a" ]; then
        echo "🎉 编译可能已完成！"
        echo "文件详情:"
        ls -la "$ONNX_DIR/build/Android/Release/"*.a 2>/dev/null | head -5
    else
        echo "💡 建议重新启动编译"
    fi
fi

echo -e "\n⏰ 检查时间: $(date)"
echo "=======================================" 