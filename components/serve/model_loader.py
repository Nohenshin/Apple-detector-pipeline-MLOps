import torch
from common.model_utils import get_model

_model = None
_device = None

def load_model(model_path, device=None):
    global _model, _device
    _device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model = get_model(num_classes=3, pretrained=False)
    _model.load_state_dict(torch.load(model_path, map_location=_device))
    _model.to(_device).eval()
    print(f'Model loaded on {_device}')

def predict(image_tensor, score_threshold=0.5):
    global _model, _device
    if _model is None:
        raise RuntimeError('Model not loaded')
    with torch.no_grad():
        pred = _model([image_tensor.to(_device)])[0]
    mask = pred['scores'] > score_threshold
    return {
        'boxes': pred['boxes'][mask].cpu().numpy().tolist(),
        'labels': pred['labels'][mask].cpu().numpy().tolist(),
        'scores': pred['scores'][mask].cpu().numpy().tolist()
    }