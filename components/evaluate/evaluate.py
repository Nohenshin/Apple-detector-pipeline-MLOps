import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from common.dataset import VOCDataset
from common.model_utils import get_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/app/data')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating on {device}')

    transform = T.ToTensor()
    test_dataset = VOCDataset(os.path.join(args.data_dir, 'test'), transforms=transform)

    def collate_fn(batch):
        return tuple(zip(*batch))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = get_model(num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    metric = MeanAveragePrecision(iou_type='bbox', class_metrics=True)
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            preds = model(images)
            metric.update(preds, targets)

    results = metric.compute()
    print(f'mAP@0.5: {results["map_50"]:.4f}')
    print(f'mAR@100: {results["mar_100"]:.4f}')
    # Per-class
    class_names = {1: 'Green Apple', 2: 'Red Apple'}
    for i, ap in enumerate(results['map_per_class']):
        class_id = results['classes'][i].item()
        name = class_names.get(class_id, f'Class {class_id}')
        print(f'{name}: AP = {ap:.4f}')

if __name__ == '__main__':
    main()