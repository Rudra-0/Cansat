"""
TinyML Aerial Landing Safety Vision Model Training Pipeline
Cognitive CanSat & TinyML Vision Platform
-----------------------------------------------------------
Hardware-Agnostic Deep Learning Trainer:
- Automatically detects and leverages NVIDIA CUDA GPUs (Tensor Cores / Mixed Precision)
- Falls back gracefully to multi-core CPU execution
- Trains a compact Depthwise Separable CNN (TinyLandingNet, ~22k params) on 27,000 EuroSAT images
- Classifies 4 Safe Landing Area Index (SLAI) tiers:
    0: SAFE_LZ           (AnnualCrop, HerbaceousVegetation, Pasture, PermanentCrop)
    1: OBSTACLE_CANOPY   (Forest)
    2: CRITICAL_HAZARD   (Highway, Industrial, Residential)
    3: WATER_HAZARD      (River, SeaLake)
- Outputs PyTorch (.pth), ONNX (.onnx), and updates ml/model_metrics.json
"""

import os
import time
import json
import random
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Path & Directory Discovery (Strictly Relative)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "eurosat", "2750")
MODELS_DIR = os.path.join(REPO_ROOT, "ml", "saved_models")
METRICS_PATH = os.path.join(REPO_ROOT, "ml", "model_metrics.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 4-Tier Tactical Landing Safety Hierarchy
# ---------------------------------------------------------------------------
SLAI_CLASSES = [
    "SAFE_LZ",          # Open fields, pastures, crops
    "OBSTACLE_CANOPY",  # Dense forests, tree canopies
    "CRITICAL_HAZARD",  # Roads, power lines, buildings, asphalt
    "WATER_HAZARD"      # Rivers, lakes, open water bodies
]

CLASS_MAP = {
    # 0: SAFE_LZ
    "AnnualCrop": 0,
    "HerbaceousVegetation": 0,
    "Pasture": 0,
    "PermanentCrop": 0,
    # 1: OBSTACLE_CANOPY
    "Forest": 1,
    # 2: CRITICAL_HAZARD
    "Highway": 2,
    "Industrial": 2,
    "Residential": 2,
    # 3: WATER_HAZARD
    "River": 3,
    "SeaLake": 3
}

# ---------------------------------------------------------------------------
# Custom PyTorch Dataset for EuroSAT with SLAI Mapping
# ---------------------------------------------------------------------------
class EuroSATLandingDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.samples: List[Tuple[str, int]] = []
        self.transform = transform
        
        candidates = [
            root_dir,
            os.path.join(root_dir, "2750"),
            os.path.join(root_dir, "EuroSAT"),
            os.path.join(os.path.dirname(root_dir), "2750"),
            os.path.join(os.path.dirname(root_dir), "EuroSAT"),
            os.path.join(REPO_ROOT, "data", "eurosat", "2750"),
            os.path.join(REPO_ROOT, "data", "eurosat", "EuroSAT"),
            os.path.join(REPO_ROOT, "data", "eurosat")
        ]
        
        found_dir = None
        for cand in candidates:
            if cand and os.path.exists(cand) and any(os.path.isdir(os.path.join(cand, c)) for c in CLASS_MAP):
                found_dir = cand
                break

        if not found_dir:
            raise FileNotFoundError(
                f"Dataset directory not found at {root_dir}. "
                "Please run: python ml/download_dataset.py"
            )

        for folder_name in sorted(os.listdir(found_dir)):
            folder_path = os.path.join(found_dir, folder_name)
            if os.path.isdir(folder_path) and folder_name in CLASS_MAP:
                target_label = CLASS_MAP[folder_name]
                for fname in os.listdir(folder_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                        self.samples.append((os.path.join(folder_path, fname), target_label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid image files found in {found_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ---------------------------------------------------------------------------
# TinyLandingNet: Ultra-Compact Depthwise Separable CNN for ESP32-CAM
# ---------------------------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class TinyLandingNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # Input: 64x64x3 RGB
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), # -> 32x32x16
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )
        self.stage1 = DepthwiseSeparableConv(16, 32, stride=2)  # -> 16x16x32
        self.stage2 = DepthwiseSeparableConv(32, 48, stride=2)  # -> 8x8x48
        self.stage3 = DepthwiseSeparableConv(48, 64, stride=2)  # -> 4x4x64
        self.gap = nn.AdaptiveAvgPool2d((1, 1))                 # -> 1x1x64
        self.dropout = nn.Dropout(p=0.2)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.head(x)

# ---------------------------------------------------------------------------
# Training Engine
# ---------------------------------------------------------------------------
def train_model(data_dir: str = None, epochs: int = 30, batch_size: int = 64, learning_rate: float = 1e-3):
    # Set deterministic seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if data_dir is None:
        dataset_source = DATA_DIR
    elif not os.path.isabs(data_dir):
        dataset_source = os.path.join(REPO_ROOT, data_dir)
    else:
        dataset_source = data_dir

    # Hardware Detection
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    
    print("=" * 70)
    print("   TINYML AERIAL LANDING SAFETY VISION MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"Hardware Compute Device : {device.type.upper()}")
    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"NVIDIA GPU Acceleration : {gpu_name} ({gpu_mem:.2f} GB VRAM)")
        torch.backends.cudnn.benchmark = True
    else:
        print("Running in CPU Mode. (CUDA will activate automatically on NVIDIA systems)")
    print("Target Architecture     : TinyLandingNet (Depthwise Separable CNN)")
    print(f"Output Tiers            : {len(SLAI_CLASSES)} ({', '.join(SLAI_CLASSES)})")
    print("-" * 70)

    # Data Transforms with Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"[1/5] Loading EuroSAT dataset from: {dataset_source} ...")
    full_dataset = EuroSATLandingDataset(dataset_source, transform=None)
    total_len = len(full_dataset)
    print(f"[OK] Found {total_len:,} labeled aerial images.")

    # 70% Train, 15% Val, 15% Test
    train_len = int(0.70 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    train_data, val_data, test_data = random_split(
        full_dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
    )

    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

    train_loader = DataLoader(
        TransformedSubset(train_data, train_transform),
        batch_size=batch_size, shuffle=True, pin_memory=is_cuda, num_workers=0
    )
    val_loader = DataLoader(
        TransformedSubset(val_data, test_transform),
        batch_size=batch_size, shuffle=False, pin_memory=is_cuda, num_workers=0
    )
    test_loader = DataLoader(
        TransformedSubset(test_data, test_transform),
        batch_size=batch_size, shuffle=False, pin_memory=is_cuda, num_workers=0
    )

    print(f"[2/5] Initializing TinyLandingNet neural network on {device}...")
    model = TinyLandingNet(num_classes=len(SLAI_CLASSES)).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model Parameters: {param_count:,} (Expected INT8 footprint: ~{param_count/1024:.1f} KB)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)

    print(f"\n[3/5] Commencing training across {epochs} epochs (Batch size: {batch_size})...")
    start_time = time.time()

    best_val_acc = 0.0
    best_weights_path = os.path.join(MODELS_DIR, "tinylandingnet_best.pth")
    legacy_weights_path = os.path.join(MODELS_DIR, "tinyml_landing_safety.pth")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=is_cuda):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=is_cuda):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= total_val
        val_acc = (correct_val / total_val) * 100.0
        epoch_dur = time.time() - epoch_start

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_dur:4.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            torch.save(model.state_dict(), legacy_weights_path)

    total_dur = time.time() - start_time
    print(f"\n[OK] Training completed in {total_dur:.1f}s. Best Val Accuracy: {best_val_acc:.2f}%")

    # -----------------------------------------------------------------------
    # Step 4: Full Test Evaluation & Confusion Matrix
    # -----------------------------------------------------------------------
    print(f"\n[4/5] Evaluating optimal model on holdout test set (N = {test_len:,})...")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_accuracy = (np.sum(all_preds == all_targets) / len(all_targets)) * 100.0

    # Build Confusion Matrix
    num_classes = len(SLAI_CLASSES)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    print("\n" + "=" * 65)
    print(f" TEST EVALUATION RESULTS: OVERALL ACCURACY = {test_accuracy:.2f}%")
    print("=" * 65)
    print(f"{'Class Name':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 65)

    metrics_per_class = {}
    f1_list = []

    for c_idx, c_name in enumerate(SLAI_CLASSES):
        tp = cm[c_idx, c_idx]
        fp = np.sum(cm[:, c_idx]) - tp
        fn = np.sum(cm[c_idx, :]) - tp
        support = int(np.sum(cm[c_idx, :]))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_list.append(f1)

        metrics_per_class[c_name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "support": support
        }
        print(f"{c_name:<20} {precision*100:6.2f}%     {recall*100:6.2f}%     {f1*100:6.2f}%     {support:<8}")

    macro_f1 = float(np.mean(f1_list)) * 100.0
    print("-" * 65)
    print(f"{'Macro Average':<20} {'':<12} {'':<12} {macro_f1:6.2f}%     {len(all_targets):<8}")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # Step 5: Export Artifacts (ONNX + JSON Metrics)
    # -----------------------------------------------------------------------
    print(f"\n[5/5] Exporting ONNX graph and updating {METRICS_PATH}...")
    onnx_path = os.path.join(MODELS_DIR, "tinylandingnet.onnx")
    legacy_onnx_path = os.path.join(MODELS_DIR, "tinyml_landing_safety.onnx")
    dummy_input = torch.randn(1, 3, 64, 64, device=device)
    
    try:
        import shutil
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input_image'],
            output_names=['slai_logits'],
            dynamic_axes={'input_image': {0: 'batch_size'}, 'slai_logits': {0: 'batch_size'}}
        )
        shutil.copyfile(onnx_path, legacy_onnx_path)
        print(f"[OK] ONNX model saved to: {onnx_path} and {legacy_onnx_path}")
    except Exception as onnx_err:
        print(f"[!] Note on ONNX export: {onnx_err}")

    # Update model_metrics.json
    metrics_data = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = {}

    metrics_data["tinyml_landing_safety_model"] = {
        "architecture": "TinyLandingNet (Depthwise Separable CNN)",
        "parameter_count": param_count,
        "input_resolution": "64x64 RGB",
        "device_trained_on": str(device).upper(),
        "test_accuracy_pct": round(float(test_accuracy), 2),
        "macro_f1_pct": round(float(macro_f1), 2),
        "classes": SLAI_CLASSES,
        "class_metrics": metrics_per_class,
        "confusion_matrix": cm.tolist(),
        "last_trained_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[OK] Updated {METRICS_PATH} successfully.")
    print("\n[SUCCESS] TinyML Aerial Vision Pipeline training complete!")

# ---------------------------------------------------------------------------
# Multimodal & Spatial Vision Utilities: 3x3 Grid, Escape Vector, VARI & TTI
# ---------------------------------------------------------------------------
def compute_vari(image: np.ndarray) -> Tuple[float, str]:
    """
    Computes the Visible Atmospheric Resistant Index (VARI) from an RGB image:
        VARI = (Green - Red) / (Green + Red - Blue)
    Returns: (vari_score, vegetation_classification)
    """
    if image is None or image.size == 0:
        return 0.0, "UNKNOWN"
    
    # Convert BGR to RGB if needed (assuming RGB input)
    arr = image.astype(np.float32)
    r = np.mean(arr[:, :, 0])
    g = np.mean(arr[:, :, 1])
    b = np.mean(arr[:, :, 2])
    
    denom = (g + r - b)
    if abs(denom) < 1e-6:
        vari = 0.0
    else:
        vari = float((g - r) / denom)
    vari = max(-1.0, min(1.0, vari))
    
    if vari > 0.20:
        label = "DENSE VEGETATION"
    elif vari >= 0.0:
        label = "MODERATE VEGETATION / CROPS"
    elif vari >= -0.15:
        label = "DRY SOIL / BARE GROUND"
    else:
        label = "WATER / ARTIFICIAL SURFACE"
        
    return round(vari, 3), label


def evaluate_3x3_spatial_grid(model: nn.Module, pil_image: Image.Image, device=None) -> Dict:
    """
    Partitions an incoming aerial image into a 3x3 spatial grid (9 sectors),
    evaluates each sector with TinyLandingNet, and computes the Directional Escape Vector.
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    w, h = pil_image.size
    cell_w = w // 3
    cell_h = h // 3
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    grid_results = []
    crops = []
    
    for r in range(3):
        row_res = []
        for c in range(3):
            box = (c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h)
            crop = pil_image.crop(box)
            tensor = transform(crop).unsqueeze(0).to(device)
            crops.append((r, c, tensor))
            
    # Batch evaluate all 9 sectors
    batch_tensors = torch.cat([t for _, _, t in crops], dim=0)
    with torch.no_grad():
        logits = model(batch_tensors)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
    scores = np.zeros((3, 3), dtype=np.float32)
    class_grid = [[None]*3 for _ in range(3)]
    
    for idx, (r, c, _) in enumerate(crops):
        p = probs[idx]
        pred_class_idx = int(np.argmax(p))
        # Safety score: Safe(+1.0), Canopy(-0.5), Critical(-1.0), Water(-1.0)
        safety_score = float(1.0 * p[0] - 0.5 * p[1] - 1.0 * p[2] - 1.0 * p[3])
        scores[r, c] = safety_score
        class_grid[r][c] = {
            "class_id": pred_class_idx,
            "class_name": SLAI_CLASSES[pred_class_idx],
            "confidence": round(float(p[pred_class_idx]), 3),
            "safety_score": round(safety_score, 3)
        }
        
    # Best evacuation sector
    best_idx = np.unravel_index(np.argmax(scores), scores.shape)
    best_r, best_c = int(best_idx[0]), int(best_idx[1])
    
    # Escape vector from frame center (1, 1)
    dx = best_c - 1 # -1 (Left), 0 (Center), +1 (Right)
    dy = -(best_r - 1) # +1 (Up), 0 (Center), -1 (Down) in cartesian
    
    heading_deg = 0.0
    if dx != 0 or dy != 0:
        heading_deg = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
        
    center_safe = (class_grid[1][1]["class_name"] == "SAFE_LZ")
    
    return {
        "grid": class_grid,
        "safest_sector": {"row": best_r, "col": best_c, "score": float(scores[best_r, best_c])},
        "escape_vector": {
            "dx": dx,
            "dy": dy,
            "heading_deg": round(float(heading_deg), 1),
            "requires_evasion": not center_safe
        }
    }


def compute_visual_tti(prev_crop: np.ndarray, curr_crop: np.ndarray, dt: float) -> float:
    """
    Estimates visual Time-To-Impact (TTI) via pixel divergence rate.
    """
    if dt <= 0 or prev_crop is None or curr_crop is None:
        return 0.0
    # Basic structural difference expansion estimate
    h, w = curr_crop.shape[:2]
    diff = np.abs(curr_crop.astype(np.float32) - prev_crop.astype(np.float32))
    divergence = np.mean(diff)
    if divergence < 1.0:
        return 99.9
    # TTI proxy scaled to nominal 5 m/s parachute descent
    tti = max(1.0, min(120.0, (255.0 / divergence) * dt * 3.5))
    return round(float(tti), 1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TinyLandingNet Aerial Safety Vision Model Trainer")
    parser.add_argument("--data-dir", type=str, default="data/eurosat/EuroSAT", help="Path to EuroSAT dataset directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    args = parser.parse_args()

    train_model(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)

