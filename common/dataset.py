# -*- coding: utf-8 -*-
# common/dataset.py

import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=None, class_dict=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.class_dict = class_dict or {"Green Apple": 1, "Red Apple": 2}

        # Ảnh và XML nằm chung thư mục root_dir
        self.images_dir = root_dir  # Sửa: bỏ os.path.join(root_dir, 'images')
        self.labels_dir = root_dir  # Sửa: bỏ os.path.join(root_dir, 'labels')

        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        xml_file = img_file.replace('.jpg', '.xml')

        img_path = os.path.join(self.images_dir, img_file)
        xml_path = os.path.join(self.labels_dir, xml_file)

        img = Image.open(img_path).convert('RGB')

        boxes = []
        labels = []
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                label = self.class_dict.get(name, 0)
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        # Xử lý trường hợp không có box nào (ảnh không có annotation)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)  # Sửa: tránh lỗi reshape
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.image_files)