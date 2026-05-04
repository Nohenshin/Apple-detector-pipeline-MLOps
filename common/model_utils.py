# -*- coding: utf-8 -*-
# common/model_utils.py
# Hàm tạo mô hình Faster R-CNN.

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes=3, pretrained=True):
    """
    Tạo mô hình Faster R-CNN với backbone ResNet50 FPN.
    num_classes: số lớp (bao gồm background). Ví dụ: 3 = background + Green Apple + Red Apple.
    pretrained: nếu True, dùng trọng số đã huấn luyện trên COCO.
    """
    if pretrained:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # Thay đổi đầu phân loại
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model