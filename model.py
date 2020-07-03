from lib.faster_rcnn import fasterrcnn_resnet_fpn
from torchvision.models.detection.rpn import AnchorGenerator
import torch
import torchvision


# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
#                                    aspect_ratios=((0.5, 1.0, 2.0),))
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
model = fasterrcnn_resnet_fpn(num_classes=91)
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)