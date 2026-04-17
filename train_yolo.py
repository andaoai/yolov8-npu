"""
YOLOv8 NPU 训练脚本
解决 _foreach_norm 在 NPU 上不支持的问题
解决 Ultralytics trainer 中 CUDA 硬编码问题
"""
import gc
import torch
import torch_npu
from ultralytics import YOLO


def clip_grad_norm_npu(parameters, max_norm: float, norm_type: float = 2.0):
    """
    NPU 兼容的梯度裁剪函数
    替代 torch.nn.utils.clip_grad_norm_，避免使用 _foreach_norm
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.)

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max().item() for p in parameters)
    elif norm_type == float('-inf'):
        total_norm = min(p.grad.data.abs().min().item() for p in parameters)
    else:
        total_norm = 0.
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)

    return torch.tensor(total_norm)


# Monkey Patch: 替换 clip_grad_norm_ 为 NPU 兼容版本
torch.nn.utils.clip_grad_norm_ = clip_grad_norm_npu


def _get_memory_npu(trainer, fraction=False):
    """NPU 兼容的内存获取方法"""
    memory, total = 0, 0
    if trainer.device.type == "mps":
        memory = torch.mps.driver_allocated_memory()
        if fraction:
            return __import__("psutil").virtual_memory().percent / 100
    elif trainer.device.type == "npu":
        memory = torch.npu.memory_allocated()
        if fraction:
            total = torch.npu.get_device_properties(trainer.device).total_memory
    elif trainer.device.type != "cpu":
        memory = torch.cuda.memory_reserved()
        if fraction:
            total = torch.cuda.get_device_properties(trainer.device).total_memory
    return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)


def _clear_memory_npu(trainer, threshold=None):
    """NPU 兼容的内存清理方法"""
    if threshold:
        assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
        if _get_memory_npu(trainer, fraction=True) <= threshold:
            return
    gc.collect()
    if trainer.device.type == "mps":
        torch.mps.empty_cache()
    elif trainer.device.type == "cpu":
        return
    elif trainer.device.type == "npu":
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


def patch_trainer(trainer):
    """替换 trainer 的内存管理方法为 NPU 兼容版本"""
    trainer._get_memory = lambda fraction=False: _get_memory_npu(trainer, fraction)
    trainer._clear_memory = lambda threshold=None: _clear_memory_npu(trainer, threshold)


def main():
    # 设置 NPU 设备
    device = "npu:0"
    print(f"使用设备: {device}")
    print(f"NPU 数量: {torch.npu.device_count()}")

    # 加载模型
    model = YOLO("yolov8n.pt")

    # 注册回调：在训练开始时 patch trainer
    model.add_callback("on_train_start", patch_trainer)

    # 开始训练
    results = model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        workers=8,
        project="runs/train",
        name="yolov8n_npu",
        exist_ok=True,
        verbose=True,
        amp=False,  # 禁用 AMP，避免 NPU 上的 CUDA 检查
    )

    print("训练完成!")
    print(f"最佳模型路径: {results.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
