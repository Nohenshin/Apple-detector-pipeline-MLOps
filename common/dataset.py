# -*- coding: utf-8 -*-
# common/dataset.py
# Lớp VOCDataset cho cấu trúc thư mục:
#   root/
#     images/   (chứa các file .jpg)
#     labels/   (chứa các file .xml cùng tên)

import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=None, class_dict=None):
        """
        Args:
            root_dir (str): Đường dẫn đến thư mục chứa images/ và labels/
                            Ví dụ: './data/train'
            transforms (callable, optional): Phép biến đổi ảnh (vd: ToTensor)
            class_dict (dict, optional): Ánh xạ tên lớp -> số nhãn
        """
        self.root_dir = root_dir
        self.transforms = transforms
        if class_dict is None:
            self.class_dict = {"Green Apple": 1, "Red Apple": 2}
        else:
            self.class_dict = class_dict

        # Đường dẫn đến thư mục con
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')

        # Lấy danh sách tất cả file .jpg trong images_dir
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        self.image_files.sort()  # Đảm bảo thứ tự nhất quán

    def __getitem__(self, idx):
        # Lấy tên file ảnh
        img_file = self.image_files[idx]
        # Suy ra file xml tương ứng (đổi .jpg -> .xml)
        xml_file = img_file.replace('.jpg', '.xml')

        img_path = os.path.join(self.images_dir, img_file)
        xml_path = os.path.join(self.labels_dir, xml_file)

        # Đọc ảnh
        img = Image.open(img_path).convert('RGB')

        # Đọc file XML nếu tồn tại
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

        # Chuyển sang tensor
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