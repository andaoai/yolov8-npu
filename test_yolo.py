"""
测试 YOLOv8 在 NPU 上的运行（优化版）
"""
import torch
import torch_npu
import time
import os

print(f"PyTorch version: {torch.__version__}")
print(f"torch_npu version: {torch_npu.__version__}")
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU count: {torch.npu.device_count()}")

if torch.npu.is_available():
    print(f"\n使用 NPU: {torch.npu.get_device_name(0)}")

    # 设置 NPU 为默认设备
    torch.npu.set_device(0)

    # 测试基本张量操作
    x = torch.randn(3, 3).npu()
    print(f"\n测试张量创建: {x.device}")

    # 加载 YOLOv8 模型
    print("\n加载 YOLOv8 模型...")
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    print("模型加载成功!")

    test_image = "test.jpg"
    print(f"\n使用图片: {test_image}")

    # 第一次推理（预热，会编译算子）
    print("\n第一次推理（预热）...")
    start = time.time()
    results = model.predict(
        source=test_image,
        device="npu:0",
        verbose=False
    )
    first_time = time.time() - start
    print(f"第一次推理时间: {first_time:.2f}s")

    # 第二次推理（应该更快）
    print("\n第二次推理...")
    start = time.time()
    results = model.predict(
        source=test_image,
        device="npu:0",
        verbose=False
    )
    second_time = time.time() - start
    print(f"第二次推理时间: {second_time:.2f}s")

    # 第三次推理
    print("\n第三次推理...")
    start = time.time()
    results = model.predict(
        source=test_image,
        device="npu:0",
        verbose=False
    )
    third_time = time.time() - start
    print(f"第三次推理时间: {third_time:.2f}s")

    print(f"\n{'='*50}")
    print("推理结果:")
    print(f"{'='*50}")
    print(f"检测到 {len(results[0].boxes)} 个目标")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        conf = float(box.conf[0])
        print(f"  - {cls_name}: {conf:.2f}")

    print(f"\n性能总结:")
    print(f"  首次推理: {first_time:.2f}s (包含算子编译)")
    print(f"  第二次推理: {second_time:.2f}s")
    print(f"  第三次推理: {third_time:.2f}s")

else:
    print("NPU 不可用，请检查环境配置")
