# YOLOv8 训练 - 昇腾 NPU 版本

本项目用于在华为昇腾 NPU (Ascend910ProB) 上运行和训练 YOLOv8 目标检测模型。

## 环境信息

| 组件 | 版本 |
|------|------|
| 操作系统 | Huawei Cloud EulerOS 2.0 (aarch64) |
| NPU | Ascend910ProB × 8 |
| 驱动版本 | 24.1.0 |
| CANN | 9.0.0-beta.2 |
| Python | 3.10 |
| PyTorch | 2.5.1 |
| torch-npu | 2.5.1.post1 |
| ultralytics | 8.4.37 |

## 快速开始

### 1. 环境准备

```bash
# 进入项目目录
cd /home/ht/test/yolov8-npu

# 切换到 HwHiAiUser 组（重要！）
newgrp HwHiAiUser

# 设置 CANN 环境变量
source /usr/local/Ascend/cann/set_env.sh
```

### 2. 安装依赖

```bash
# 安装 PyTorch 和 torch-npu
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 torch-npu==2.5.1.post1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 ultralytics 和其他依赖
.venv/bin/pip install ultralytics opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 CANN 相关依赖
.venv/bin/pip install decorator attrs cloudpickle psutil scipy tornado pyyaml ml-dtypes -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 验证环境

```bash
# 检查 NPU 是否可用
.venv/bin/python -c "
import torch
import torch_npu
print(f'PyTorch: {torch.__version__}')
print(f'torch_npu: {torch_npu.__version__}')
print(f'NPU available: {torch.npu.is_available()}')
print(f'NPU count: {torch.npu.device_count()}')
"
```

### 4. 运行推理测试

```bash
./test_npu_env.sh
```

## 完整安装步骤（从零开始）

### 步骤 1: 创建用户组（如已存在可跳过）

```bash
sudo groupadd HwHiAiUser
sudo usermod -aG HwHiAiUser ht
```

### 步骤 2: 配置 Ascend 软件源

```bash
sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo
sudo yum makecache
```

### 步骤 3: 安装系统依赖

```bash
sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
```

### 步骤 4: 安装 NPU 驱动

```bash
sudo yum install -y Ascend-hdk-910-npu-driver-25.5.0
```

### 步骤 5: 安装 CANN Toolkit

```bash
sudo yum install -y Ascend-cann-toolkit-9.0.0_beta.2
sudo yum install -y Ascend-cann-910-ops-9.0.0_beta.2
```

### 步骤 6: 验证安装

```bash
source /usr/local/Ascend/cann/set_env.sh
python3 -c "import acl;print(acl.get_soc_name())"
# 输出: Ascend910ProB
```

### 步骤 7: 创建 Python 虚拟环境

```bash
uv venv --python 3.10
.venv/bin/pip install pip -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 使用方法

### 推理

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.pt")

# 在 NPU 上推理
results = model.predict(
    source="test.jpg",
    device="npu:0",
    save=True
)
```

### 训练

```bash
# 检查环境
.venv/bin/python main.py --check

# 创建数据集配置
.venv/bin/python main.py --create-config

# 开始训练
.venv/bin/python main.py --data dataset.yaml --epochs 100 --batch 16 --device npu:0

# 多卡训练
.venv/bin/python main.py --data dataset.yaml --device npu:0,1,2,3
```

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 数据集配置文件路径 | `dataset.yaml` |
| `--model` | 模型大小 (n/s/m/l/x) | `n` |
| `--epochs` | 训练轮数 | `100` |
| `--batch` | 批次大小 | `16` |
| `--img-size` | 输入图像尺寸 | `640` |
| `--device` | 训练设备 | `npu:0` |
| `--resume` | 恢复训练的检查点 | - |

## 数据集格式

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image3.jpg
│       └── image4.jpg
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        ├── image3.txt
        └── image4.txt
```

标签文件格式 (YOLO):
```
class_id center_x center_y width height
```

## 项目结构

```
yolov8-npu/
├── main.py              # 训练脚本
├── test_yolo.py         # 推理测试脚本
├── test_npu_env.sh      # 环境测试脚本（设置环境变量并运行推理测试）
├── install_cann.sh      # CANN 一键安装脚本（需 sudo 执行）
├── dataset.yaml         # 数据集配置模板
├── pyproject.toml       # 项目依赖
├── test.jpg             # 测试图片
└── .venv/               # Python 虚拟环境
```

## 脚本说明

### test_npu_env.sh - 环境测试脚本

用于验证 NPU 环境并运行 YOLOv8 推理测试。

```bash
#!/bin/bash
# 设置环境
source /usr/local/Ascend/cann/set_env.sh

# 运行测试
/home/ht/test/yolov8-npu/.venv/bin/python test_yolo.py
```

**使用方法：**
```bash
newgrp HwHiAiUser
./test_npu_env.sh
```

### install_cann.sh - CANN 一键安装脚本

自动完成 CANN 环境的安装配置，包括：
- 添加用户到 HwHiAiUser 组
- 配置 Ascend 软件源
- 安装系统依赖
- 安装 NPU 驱动
- 安装 CANN Toolkit 和算子库

```bash
#!/bin/bash
# CANN 安装脚本
# 使用方法: sudo bash install_cann.sh

set -e

echo "=========================================="
echo "CANN 安装脚本"
echo "=========================================="

# 1. 将 ht 用户添加到 HwHiAiUser 组
echo ""
echo "[1/5] 将 ht 用户添加到 HwHiAiUser 组..."
usermod -aG HwHiAiUser ht
echo "完成!"

# 2. 配置 Ascend 源
echo ""
echo "[2/5] 配置 Ascend 软件源..."
curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo
yum makecache
echo "完成!"

# 3. 安装依赖
echo ""
echo "[3/5] 安装依赖包..."
yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
echo "完成!"

# 4. 安装 NPU 驱动 (如果需要更新)
echo ""
echo "[4/5] 检查 NPU 驱动..."
if rpm -q Ascend-hdk-910-npu-driver &>/dev/null; then
    echo "NPU 驱动已安装: $(rpm -q Ascend-hdk-910-npu-driver)"
else
    echo "安装 NPU 驱动..."
    yum install -y Ascend-hdk-910-npu-driver-25.5.0
fi
echo "完成!"

# 5. 安装 CANN Toolkit
echo ""
echo "[5/5] 安装 CANN Toolkit 和算子库..."
yum install -y Ascend-cann-toolkit-9.0.0_beta.2
yum install -y Ascend-cann-910-ops-9.0.0_beta.2
echo "完成!"

echo ""
echo "=========================================="
echo "安装完成!"
echo "=========================================="
echo ""
echo "请执行以下步骤完成配置:"
echo ""
echo "1. 重新登录或执行: newgrp HwHiAiUser"
echo ""
echo "2. 配置环境变量:"
echo "   source /usr/local/Ascend/cann/set_env.sh"
echo ""
echo "3. 验证安装:"
echo "   python3 -c \"import acl;print(acl.get_soc_name())\""
echo ""
```

**使用方法：**
```bash
sudo bash install_cann.sh
```

## 性能优化建议

1. **首次推理较慢** - 第一次运行会编译算子，后续会快很多
2. **使用更大的模型** - NPU 对大模型（yolov8s/m/l）效率更高
3. **增大 batch size** - 充分利用 NPU 并行能力
4. **NMS 回退 CPU** - `torchvision::nms` 目前不支持 NPU，会回退到 CPU

## 常见问题

### 1. ImportError: libhccl.so: cannot open shared object file

需要设置 CANN 环境变量:
```bash
source /usr/local/Ascend/cann/set_env.sh
```

### 2. NPU 不可用 / NPU count: 0

检查用户权限:
```bash
# 确保用户在 HwHiAiUser 组
groups ht

# 如果不在，添加用户到组
sudo usermod -aG HwHiAiUser ht

# 重新登录或执行
newgrp HwHiAiUser
```

### 3. RuntimeError: Failed to load the backend extension

检查 torch 和 torch-npu 版本兼容性:
```bash
# 正确的版本组合
torch==2.5.1
torch-npu==2.5.1.post1
```

### 4. ImportError: libGL.so.1: cannot open shared object file

使用 headless 版本的 OpenCV:
```bash
.venv/bin/pip uninstall opencv-python -y
.venv/bin/pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5. ModuleNotFoundError: No module named 'decorator'

安装缺失的 CANN 依赖:
```bash
.venv/bin/pip install decorator attrs cloudpickle psutil scipy tornado pyyaml ml-dtypes -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 参考资料

- [昇腾官方文档](https://www.hiascend.com/document)
- [CANN 下载](https://www.hiascend.com/software/cann)
- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [PyTorch 昇腾适配](https://gitee.com/ascend/pytorch)

## License

MIT
