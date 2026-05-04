# -*- coding: utf-8 -*-
# common/model_utils.py
# Hàm tạo mô hình Faster R-CNN.

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

def get_model(num_classes=3, pretrained=True):
    """
    Tạo mô hình Faster R-CNN với backbone ResNet50 FPN.
    num_classes: số lớp (bao gồm background). Ví dụ: 3 = background + Green Apple + Red Apple.
    pretrained: nếu True, dùng trọng số đã huấn luyện trên COCO.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model