import os
import sys
import json
import torch
from easydict import EasyDict as edict
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import cv2
from flask import Flask, request, render_template, send_from_directory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from model.classifier import Classifier
from data.utils import transform

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Disease classes and explanations
disease_classes = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
disease_explanations = {
    'Cardiomegaly': "Cardiomegaly means an enlargement of the heart. It can be a sign of heart conditions that need medical attention.",
    'Edema': "Edema refers to swelling caused by excess fluid in the body’s tissues, often in the lungs in this context.",
    'Consolidation': "Consolidation happens when air in the lungs is replaced with fluid or other material, often due to infection like pneumonia.",
    'Atelectasis': "Atelectasis is a collapse of part or all of a lung, which can make breathing harder.",
    'Pleural Effusion': "Pleural Effusion is a buildup of fluid between the lung and chest wall, which can cause discomfort or breathing issues."
}

# Load model and config at startup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg_path = './config/cfg.json'
ckpt_path = './config/pre_train.pth'

print("Loading config and model...")
with open(cfg_path) as f:
    cfg = edict(json.load(f))

model = Classifier(cfg)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt)
model.to(device)
print("Model loaded successfully.")

# Existing functions (unchanged)
def tensor2numpy(tensor):
    return tensor.cpu().detach().numpy()

def fig2data(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    buf = buf[:, :, :3]
    buf = buf[:, :, [2, 1, 0]]
    return np.ascontiguousarray(buf)

def image_reader(image_file, cfg):
    print(f"Reading image: {image_file}")
    image_gray = cv2.imread(image_file, 0)
    assert image_gray is not None, f"Invalid image read: {image_file}"
    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    image = transform(image_gray, cfg)
    image = torch.from_numpy(image).unsqueeze(0)
    return image, image_color

def generate_heatmap(image_file, model, cfg, alpha, device, output_dir):
    print(f"Generating heatmap for {image_file}")
    image_tensor, image_color = image_reader(image_file, cfg)
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_(True)

    model.eval()
    feat_map = None
    def hook_fn(module, input, output):
        nonlocal feat_map
        feat_map = output
        feat_map.retain_grad()
    handle = model.backbone.register_forward_hook(hook_fn)

    logits, _ = model(image_tensor)
    logits = torch.stack(logits)

    heatmaps = []
    num_tasks = len(disease_classes)
    for i in range(num_tasks):
        model.zero_grad()
        score = logits[i]
        score.backward(retain_graph=True)
        gradients = feat_map.grad
        if gradients is None:
            raise RuntimeError("Gradients not computed for feat_map.")
        weights = F.adaptive_avg_pool2d(gradients, 1)
        heatmap = torch.sum(weights * feat_map, dim=1).squeeze(0)
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmaps.append(tensor2numpy(heatmap))

    handle.remove()

    image_np = tensor2numpy(image_tensor)[0, 0, :, :]
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
    return output_path

def get_transform(cfg):
    return transforms.Compose([
        transforms.Resize((cfg.height, cfg.width)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[cfg.pixel_mean/255.0]*3, std=[cfg.pixel_std/255.0]*3)
    ])

def get_pred(output, cfg):
    if cfg.criterion in ['BCE', 'FL']:
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

def process_image(image_path, alpha=0.2):
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert('L')
    transform_fn = get_transform(cfg)
    image_tensor = transform_fn(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        logits = output[0]
        num_tasks = len(cfg.num_classes)
        pred = np.zeros(num_tasks)
        for i in range(num_tasks):
            pred[i] = get_pred(logits[i], cfg)

    heatmap_path = generate_heatmap(image_path, model, cfg, alpha, device, app.config['OUTPUT_FOLDER'])
    
    predictions = []
    for label, prob in zip(disease_classes, pred):
        prob_percent = prob * 100
        explanation = disease_explanations[label]
        if prob_percent > 50:
            message = f"Yes, there’s a {prob_percent:.1f}% chance you have this. This is a high likelihood, so please consult a doctor."
        elif prob_percent < 10:
            message = f"No, there’s only a {prob_percent:.1f}% chance you have this. This is very low, so it’s unlikely."
        else:
            message = f"There’s a {prob_percent:.1f}% chance you have this. This is moderate, so it might be worth checking with a doctor."
        predictions.append({
            'label': label,
            'probability': f"{prob:.4f}",
            'percent': prob_percent,
            'explanation': explanation,
            'message': message
        })
    
    print("Image processed successfully.")
    return predictions, heatmap_path

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    print("Received request...")
    if request.method == 'POST':
        print("POST request detected.")
        if 'file' not in request.files:
            print("No file part in request.")
            return render_template('upload.html', error="No file part")
        file = request.files['file']
        if file.filename == '':
            print("No file selected.")
            return render_template('upload.html', error="No selected file")
        if file:
            filename = file.filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Saving file to: {image_path}")
            file.save(image_path)
            
            try:
                predictions, heatmap_path = process_image(image_path)
                heatmap_filename = os.path.basename(heatmap_path)
                print(f"Rendering result with heatmap: {heatmap_filename}")
                os.remove(image_path)
                return render_template('result.html', predictions=predictions, heatmap=heatmap_filename)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                os.remove(image_path)
                return render_template('upload.html', error=f"Error processing image: {str(e)}")
    print("Rendering upload page.")
    return render_template('upload.html')

@app.route('/output/<filename>')
def serve_heatmap(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=False)

@app.route('/download/<filename>')
def download_heatmap(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)