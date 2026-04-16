"""
YOLOv8 训练脚本 - 昇腾 NPU 版本
"""
import os
import argparse


def check_npu_environment():
    """检查 NPU 环境是否正确配置"""
    print("=" * 50)
    print("检查 NPU 环境...")
    print("=" * 50)

    # 检查环境变量
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if ascend_path:
        print(f"[OK] ASCEND_HOME_PATH: {ascend_path}")
    else:
        print("[WARNING] ASCEND_HOME_PATH 未设置")

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if "ascend" in ld_library_path.lower():
        print(f"[OK] LD_LIBRARY_PATH 包含 Ascend 路径")
    else:
        print("[WARNING] LD_LIBRARY_PATH 未包含 Ascend 路径")

    # 尝试导入 torch_npu
    try:
        import torch
        import torch_npu

        print(f"[OK] PyTorch 版本: {torch.__version__}")
        print(f"[OK] torch_npu 版本: {torch_npu.__version__}")

        if torch.npu.is_available():
            print(f"[OK] NPU 可用")
            print(f"     NPU 数量: {torch.npu.device_count()}")
            for i in range(torch.npu.device_count()):
                print(f"     NPU {i}: {torch.npu.get_device_name(i)}")
            return True
        else:
            print("[ERROR] NPU 不可用")
            return False
    except ImportError as e:
        print(f"[ERROR] 导入 torch_npu 失败: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] 检查 NPU 时出错: {e}")
        return False


def train_yolov8(
    data_config: str,
    model_size: str = "n",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "npu:0",
    resume: str = None,
):
    """
    训练 YOLOv8 模型

    Args:
        data_config: 数据集配置文件路径 (YAML)
        model_size: 模型大小 (n, s, m, l, x)
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 输入图像尺寸
        device: 训练设备 (npu:0, npu:0,1,2,3 等)
        resume: 恢复训练的检查点路径
    """
    from ultralytics import YOLO

    # 选择模型
    model_name = f"yolov8{model_size}.pt"

    if resume:
        print(f"从检查点恢复训练: {resume}")
        model = YOLO(resume)
    else:
        print(f"使用预训练模型: {model_name}")
        model = YOLO(model_name)

    # 开始训练
    print("=" * 50)
    print("开始训练 YOLOv8")
    print(f"  数据集: {data_config}")
    print(f"  模型: {model_name}")
    print(f"  轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  图像尺寸: {img_size}")
    print(f"  设备: {device}")
    print("=" * 50)

    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project="runs/train",
        name="yolov8_npu",
        exist_ok=True,
    )

    print("训练完成!")
    print(f"结果保存在: runs/train/yolov8_npu/")

    return results


def create_sample_dataset_config(output_path: str = "dataset.yaml"):
    """创建示例数据集配置文件"""
    config = """# YOLOv8 数据集配置
# 请根据你的数据集修改以下路径和类别

# 数据集根目录
path: /path/to/your/dataset

# 训练集图像目录 (相对于 path)
train: images/train

# 验证集图像目录 (相对于 path)
val: images/val

# 类别数量
nc: 2

# 类别名称
names:
  - class1
  - class2
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(config)
    print(f"示例数据集配置已创建: {output_path}")
    print("请修改配置文件中的路径和类别信息")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 训练脚本 - 昇腾 NPU 版本")

    parser.add_argument(
        "--check", action="store_true", help="检查 NPU 环境"
    )
    parser.add_argument(
        "--create-config", action="store_true", help="创建示例数据集配置文件"
    )
    parser.add_argument(
        "--data", type=str, default="dataset.yaml", help="数据集配置文件路径"
    )
    parser.add_argument(
        "--model", type=str, default="n", choices=["n", "s", "m", "l", "x"],
        help="模型大小 (n=nano, s=small, m=medium, l=large, x=xlarge)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="训练轮数"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="批次大小"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="输入图像尺寸"
    )
    parser.add_argument(
        "--device", type=str, default="npu:0", help="训练设备 (npu:0, npu:0,1,2,3)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="恢复训练的检查点路径"
    )

    args = parser.parse_args()

    if args.check:
        check_npu_environment()
        return

    if args.create_config:
        create_sample_dataset_config()
        return

    # 检查环境后训练
    if check_npu_environment():
        train_yolov8(
            data_config=args.data,
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            device=args.device,
            resume=args.resume,
        )
    else:
        print("\n" + "=" * 50)
        print("NPU 环境检查失败!")
        print("=" * 50)
        print("\n请确保已安装以下组件:")
        print("1. Ascend 驱动 (已安装: /usr/local/Ascend/driver)")
        print("2. CANN 工具包 (需要安装)")
        print("3. 设置环境变量:")
        print("   source /usr/local/Ascend/ascend-toolkit/set_env.sh")
        print("   或")
        print("   export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit")
        print("   export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH")
        print("\nCANN 下载地址:")
        print("https://www.hiascend.com/software/cann")


if __name__ == "__main__":
    main()
