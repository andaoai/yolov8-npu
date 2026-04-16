#!/bin/bash
# 设置环境
source /usr/local/Ascend/cann/set_env.sh

# 运行测试
/home/ht/test/yolov8-npu/.venv/bin/python test_yolo.py
