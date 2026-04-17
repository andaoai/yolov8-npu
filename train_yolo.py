from ultralytics import YOLO
from PIL import Image
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

torch_npu.npu.set_compile_mode(jit_compile=False)
device = torch.device('npu:{}'.format(0))

# Create a new YOLO model from scratch
# model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")
# model.to(device)

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="dataset.yaml", epochs=30, device=device, amp=False)

# Evaluate the model's performance on the validation set
results = model.val(device=device)

# Perform object detection on an image using the model
results = model.predict("./test.jpg", device=device)

# Visualize the results
# for i, r in enumerate(results):
#     # Plot results image
#     im_bgr = r.plot()  # BGR-order numpy array
#     im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

#     # Show results to screen (in supported environments)
#     r.show()

#     # Save results to disk
#     r.save(filename=f"results{i}.jpg")

# Export the model to ONNX format
# success = model.export(format="onnx")
