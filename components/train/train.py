# -*- coding: utf-8 -*-
# components/train/train.py
# Script huấn luyện mô hình. Import từ common.

import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from collections import defaultdict
import matplotlib.pyplot as plt

# Import các module từ common (đã được copy vào cùng thư mục trong Docker)
from common.dataset import VOCDataset
from common.model_utils import get_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN')
    parser.add_argument('--data_dir', type=str, default='/app/data',
                        help='Thư mục gốc chứa train/ và valid/')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--output_dir', type=str, default='/app/output',
                        help='Nơi lưu model và biểu đồ')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    os.makedirs(args.output_dir, exist_ok=True)

    # Chuẩn bị dữ liệu
    transform = T.ToTensor()
    train_dataset = VOCDataset(os.path.join(args.data_dir, 'train'), transforms=transform)
    val_dataset = VOCDataset(os.path.join(args.data_dir, 'valid'), transforms=transform)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn)

    # Mô hình
    model = get_model(num_classes=3, pretrained=True)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    best_val_map = 0.0
    early_stop_counter = 0
    early_stop_patience = 5
    train_losses = []
    val_maps = []

    for epoch in range(args.epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        loss_stats = defaultdict(float)
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
            for k, v in loss_dict.items():
                loss_stats[k] += v.item()
        mean_train_loss = epoch_loss / len(train_loader)
        train_losses.append(mean_train_loss)

        # Validation
        model.eval()
        metric = MeanAveragePrecision()
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                preds = model(images)
                metric.update(preds, targets)
        val_map = metric.compute()['map'].item()
        val_maps.append(val_map)

        print(f'Epoch {epoch+1}/{args.epochs} | Loss: {mean_train_loss:.4f} | Val mAP: {val_map:.4f}')
        scheduler.step(val_map)

        # Early stopping & save best
        if val_map > best_val_map:
            best_val_map = val_map
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print('  -> Saved best model')
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print('Early stopping triggered')
                break

    # Vẽ biểu đồ
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses)
    plt.title('Train Loss')
    plt.subplot(1,2,2)
    plt.plot(val_maps, color='green')
    plt.title('Validation mAP')
    plt.savefig(os.path.join(args.output_dir, 'training_plot.png'))
    print(f'Training completed. Best mAP: {best_val_map:.4f}')

if __name__ == '__main__':
    main()