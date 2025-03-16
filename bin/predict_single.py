import os
import sys
import argparse
import json
import torch
from easydict import EasyDict as edict
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import cv2

# Set matplotlib backend to 'Agg' before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Force Agg backend
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from model.classifier import Classifier  # noqa
from data.utils import transform  # noqa

# Disease classes
disease_classes = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']

# Argument parser
parser = argparse.ArgumentParser(description='Predict and generate heatmap for a single chest X-ray image using Grad-CAM')
parser.add_argument('--model_path', default='./', type=str,
                    help="Path to the trained models directory (containing cfg.json and pre_train.pth)")
parser.add_argument('--image_path', type=str, required=True,
                    help="Path to the input image")
parser.add_argument('--device_ids', default='0', type=str,
                    help="GPU indices comma separated, e.g. '0'")
parser.add_argument('--output_dir', default='.', type=str,
                    help="Directory to save the heatmap image (default: current directory)")
parser.add_argument('--alpha', default=0.2, type=float,
                    help="Transparency alpha of the heatmap, default 0.2")

def tensor2numpy(tensor):
    """Convert tensor to numpy array."""
    return tensor.cpu().detach().numpy()

def fig2data(fig):
    """Convert Matplotlib figure to a NumPy array using buffer_rgba."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    buf = buf[:, :, :3]  # Extract RGB, discard alpha
    buf = buf[:, :, [2, 1, 0]]  # Convert RGB to BGR for OpenCV
    return np.ascontiguousarray(buf)

def image_reader(image_file, cfg):
    """Read and preprocess image like Heatmaper."""
    image_gray = cv2.imread(image_file, 0)
    assert image_gray is not None, f"Invalid image read: {image_file}"
    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    image = transform(image_gray, cfg)
    image = torch.from_numpy(image).unsqueeze(0)
    return image, image_color

def generate_heatmap(image_file, model, cfg, alpha, device, output_dir):
    """Generate heatmap using Grad-CAM."""
    image_tensor, image_color = image_reader(image_file, cfg)
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_(True)  # Enable gradients for Grad-CAM

    model.eval()
    
    # Hook to capture feature maps from the backbone
    feat_map = None
    def hook_fn(module, input, output):
        nonlocal feat_map
        feat_map = output
        feat_map.retain_grad()  # Ensure gradients are retained for this tensor
    handle = model.backbone.register_forward_hook(hook_fn)

    # Forward pass
    logits, _ = model(image_tensor)
    logits = torch.stack(logits)  # (num_tasks, N)

    # Generate Grad-CAM heatmaps for each class
    heatmaps = []
    num_tasks = len(disease_classes)
    for i in range(num_tasks):
        model.zero_grad()
        score = logits[i]  # Target the logit for the i-th class
        score.backward(retain_graph=True)
        gradients = feat_map.grad  # (N, C, H, W)
        if gradients is None:
            raise RuntimeError("Gradients not computed for feat_map. Check model and hook setup.")
        weights = F.adaptive_avg_pool2d(gradients, 1)  # (N, C, 1, 1)
        heatmap = torch.sum(weights * feat_map, dim=1).squeeze(0)  # (H, W)
        heatmap = F.relu(heatmap)  # Positive contributions only
        heatmap = heatmap / (heatmap.max() + 1e-8)  # Normalize to [0, 1]
        heatmaps.append(tensor2numpy(heatmap))

    handle.remove()  # Remove the hook

    # Visualization
    image_np = tensor2numpy(image_tensor)[0, 0, :, :]  # (H, W)
    ori_image = image_np * cfg.pixel_std + cfg.pixel_mean
    plt_fig = plt.figure(figsize=(10, (num_tasks // 3 + 1) * 4), dpi=300)

    for i in range(num_tasks):
        prob = torch.sigmoid(logits[i]).item()
        subtitle = f'{disease_classes[i]}:{prob:.4f}'
        ax_overlay = plt_fig.add_subplot(num_tasks // 3 + 1, 3, 2 + i)
        ax_overlay.set_title(subtitle, fontsize=10, color='r')
        ax_overlay.set_yticklabels([])
        ax_overlay.set_xticklabels([])
        ax_overlay.tick_params(axis='both', which='both', length=0)
        ax_overlay.imshow(ori_image, cmap='gray', vmin=0, vmax=255)
        overlay_image = ax_overlay.imshow(cv2.resize(heatmaps[i], (cfg.long_side, cfg.long_side)),
                                         cmap='jet', vmin=0.0, vmax=1.0, alpha=alpha)

    ax_rawimage = plt_fig.add_subplot(num_tasks // 3 + 1, 3, 1)
    ax_rawimage.set_title('original image', fontsize=10, color='r')
    ax_rawimage.set_yticklabels([])
    ax_rawimage.set_xticklabels([])
    ax_rawimage.tick_params(axis='both', which='both', length=0)
    ax_rawimage.imshow(image_color)

    divider = make_axes_locatable(ax_overlay)
    ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
    plt_fig.colorbar(overlay_image, cax=ax_colorbar)
    plt_fig.tight_layout()

    figure_data = fig2data(plt_fig)
    plt.close()

    output_path = os.path.join(output_dir, f"heatmap_{os.path.basename(image_file)}")
    cv2.imwrite(output_path, figure_data)
    print(f"Heatmap saved to: {output_path}")
    return True

def get_transform(cfg):
    return transforms.Compose([
        transforms.Resize((cfg.height, cfg.width)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[cfg.pixel_mean/255.0]*3, std=[cfg.pixel_std/255.0]*3)
    ])

def get_pred(output, cfg):
    if cfg.criterion == 'BCE' or cfg.criterion == "FL":
        for num_class in cfg.num_classes:
            assert num_class == 1
        pred = torch.sigmoid(output).cpu().detach().numpy().item()
    elif cfg.criterion == 'CE':
        for num_class in cfg.num_classes:
            assert num_class >= 2
        prob = F.softmax(output, dim=1)
        pred = prob[0, 1].cpu().detach().numpy().item()
    else:
        raise Exception(f'Unknown criterion: {cfg.criterion}')
    return pred

def predict_and_heatmap(cfg, args, model):
    image = Image.open(args.image_path).convert('L')
    transform_fn = get_transform(cfg)
    image_tensor = transform_fn(image).unsqueeze(0)

    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))
        logits = output[0]  # First element of tuple is logits
        num_tasks = len(cfg.num_classes)
        pred = np.zeros(num_tasks)
        for i in range(num_tasks):
            pred[i] = get_pred(logits[i], cfg)

    print(f"Predictions for {args.image_path}:")
    for label, prob in zip(disease_classes, pred):
        print(f"{label}: {prob:.4f}")

    generate_heatmap(args.image_path, model, cfg, args.alpha, device, args.output_dir)

def main():
    global args
    args = parser.parse_args()

    cfg_path = os.path.join(args.model_path, 'cfg.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
    with open(cfg_path) as f:
        cfg = edict(json.load(f))

    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')

    model = Classifier(cfg)
    ckpt_path = os.path.join(args.model_path, 'pre_train.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)  # Load directly, no extra layers to filter
    model = model.to(device)

    predict_and_heatmap(cfg, args, model)

if __name__ == '__main__':
    main()