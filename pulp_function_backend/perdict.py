import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import os
import json
import numpy as np
from ultralytics import YOLO
import cv2
from collections import Counter

# -----------------------------
# 1️⃣ Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Using device: {device}\n")

# -----------------------------
# 2️⃣ YOLO + TTA Configuration
# -----------------------------
USE_ENSEMBLE = True  # Enable ensemble prediction (runs all 3 methods and picks best)
YOLO_MODEL_PATH = "yolov8n.pt"  # Path to YOLO model (will auto-download if not found)
YOLO_CONF_THRESHOLD = 0.25  # Confidence threshold for YOLO detections
YOLO_IOU_THRESHOLD = 0.45   # IoU threshold for NMS
TTA_STEPS = 8   # Number of augmented predictions

print(f"🎯 Ensemble Mode: {'ENABLED' if USE_ENSEMBLE else 'DISABLED'}")
if USE_ENSEMBLE:
    print(f"   Will run 3 methods and intelligently select best result")
print(f"   YOLO Model: {YOLO_MODEL_PATH}")
print(f"   TTA steps: {TTA_STEPS}\n")

# -----------------------------
# 3️⃣ Load YOLO Model
# -----------------------------
yolo_model = None
try:
    print(f"📥 Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"✅ YOLO model loaded successfully")
    print(f"   Available classes: {len(yolo_model.names)} classes\n")
except Exception as e:
    print(f"⚠️  Warning: Could not load YOLO model: {str(e)}")
    print(f"   Ensemble mode will use 2 methods instead of 3\n")

# -----------------------------
# 4️⃣ Classification Model Configuration
# -----------------------------
MODEL_PATH = "plant_disease_best.pth"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file not found — {MODEL_PATH}")
    exit(1)

# -----------------------------
# 5️⃣ Load classification model and extract metadata
# -----------------------------
print(f"📥 Loading classification model from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=device)

if 'classes' in checkpoint:
    class_names = checkpoint['classes']
    print(f"✅ Loaded {len(class_names)} classes from checkpoint")
else:
    print("⚠️  Warning: Classes not found in checkpoint, using hardcoded names")
    class_names = [
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
        "Potato___Early_blight",
        "Potato___healthy",
        "Potato___Late_blight",
        "Tomato__Tomato_mosaic_virus",
        "Tomato_Early_blight",
        "Tomato_healthy",
        "Tomato_Late_blight"
    ]

num_classes = len(class_names)

print(f"\n🏷️  Available classes:")
for idx, name in enumerate(class_names):
    print(f"  {idx}: {name}")

# -----------------------------
# 6️⃣ Build classification model
# -----------------------------
print(f"\n🏗️  Building classification model...")

model = models.efficientnet_b0(weights=None)

for param in model.features[:5].parameters():
    param.requires_grad = False

num_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, num_classes)
)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()
print("✅ Classification model loaded successfully")

if 'test_acc' in checkpoint:
    print(f"📊 Model test accuracy: {checkpoint['test_acc']:.2f}%")
if 'val_acc' in checkpoint:
    print(f"📊 Model validation accuracy: {checkpoint['val_acc']:.2f}%")

# -----------------------------
# 7️⃣ YOLO Detection Function
# -----------------------------
def detect_leaf_regions(image_path, conf_threshold=None, iou_threshold=None):
    """
    Detect leaf/plant regions using YOLO
    """
    if conf_threshold is None:
        conf_threshold = YOLO_CONF_THRESHOLD
    if iou_threshold is None:
        iou_threshold = YOLO_IOU_THRESHOLD
    
    # Load image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path
    
    # Run YOLO detection
    results = yolo_model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
    
    detected_regions = []
    
    # Extract bounding boxes
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            class_name = yolo_model.names[cls]
            
            # Crop the region
            crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
            
            detected_regions.append({
                'crop': crop,
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': conf,
                'class': class_name,
                'area': (x2 - x1) * (y2 - y1)
            })
    
    # Sort by area (largest first)
    detected_regions.sort(key=lambda x: x['area'], reverse=True)
    
    return detected_regions, image

# -----------------------------
# 8️⃣ Base Transform (for single prediction)
# -----------------------------
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# 9️⃣ Test-Time Augmentation (TTA) Transforms
# -----------------------------
def get_tta_transforms():
    """
    Returns a list of augmentation transforms for TTA
    """
    tta_transforms = []
    
    # Original (center crop)
    tta_transforms.append(transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    
    # Horizontal flip
    tta_transforms.append(transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    
    # Five crop (center + 4 corners)
    tta_transforms.append(transforms.Compose([
        transforms.Resize(256),
        transforms.FiveCrop(224),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])(crop) for crop in crops
        ]))
    ]))
    
    # Slight rotation variations
    for angle in [-10, 10]:
        tta_transforms.append(transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(degrees=(angle, angle)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    
    # Brightness adjustments
    for factor in [0.9, 1.1]:
        tta_transforms.append(transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: ImageEnhance.Brightness(img).enhance(factor)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    
    # Contrast adjustments
    tta_transforms.append(transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda img: ImageEnhance.Contrast(img).enhance(1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    
    return tta_transforms

# -----------------------------
# 🔟 TTA Prediction Function
# -----------------------------
def predict_with_tta(image, num_tta=8):
    """
    Make prediction using Test-Time Augmentation
    """
    tta_transforms = get_tta_transforms()
    all_predictions = []
    
    selected_transforms = tta_transforms[:min(num_tta, len(tta_transforms))]
    
    for transform in selected_transforms:
        try:
            transformed = transform(image)
            
            # Handle FiveCrop (which returns 5 images)
            if len(transformed.shape) == 4:
                for crop in transformed:
                    input_tensor = crop.unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        all_predictions.append(probs[0].cpu().numpy())
            else:
                input_tensor = transformed.unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    all_predictions.append(probs[0].cpu().numpy())
        except Exception as e:
            continue
    
    if not all_predictions:
        raise Exception("All TTA transforms failed")
    
    # Average all predictions
    avg_prediction = np.mean(all_predictions, axis=0)
    
    return torch.from_numpy(avg_prediction)

# -----------------------------
# 1️⃣1️⃣ Single Prediction (no TTA)
# -----------------------------
def predict_single(image):
    """
    Make single prediction without TTA
    """
    input_tensor = base_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
    return probs[0]

# -----------------------------
# 1️⃣2️⃣ Ensemble Prediction Strategy
# -----------------------------
def calculate_prediction_score(prediction_result):
    """
    Calculate a reliability score for a prediction based on multiple factors
    
    Scoring criteria:
    - Confidence level (0-100)
    - Confidence gap between top 2 predictions (higher gap = more confident)
    - Method reliability bonus (YOLO+TTA > TTA > Basic)
    """
    confidence = prediction_result['confidence']
    top_preds = prediction_result['top_k_predictions']
    method = prediction_result['method']
    
    # Base score from confidence
    score = confidence
    
    # Bonus for confidence gap (difference between 1st and 2nd prediction)
    if len(top_preds) > 1:
        conf_gap = top_preds[0]['confidence'] - top_preds[1]['confidence']
        score += conf_gap * 0.3  # 30% weight on gap
    
    # Method reliability bonus
    method_bonus = {
        'YOLO+TTA': 15,
        'TTA': 10,
        'Basic': 0
    }
    score += method_bonus.get(method, 0)
    
    # Penalty for low confidence
    if confidence < 50:
        score *= 0.7
    
    return score

def ensemble_predict(image_path, verbose=True):
    """
    Run all 3 prediction methods and intelligently select the best result
    
    Methods:
    1. Basic (no YOLO, no TTA) - Fastest but least reliable
    2. TTA only - Good for clean images
    3. YOLO + TTA - Best for images with backgrounds
    
    Selection strategy:
    - If all methods agree: Return consensus with highest confidence
    - If methods disagree: Use scoring system based on confidence, gap, and method reliability
    - Consider voting when 2/3 methods agree
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🧠 ENSEMBLE PREDICTION MODE")
        print(f"{'='*70}")
        print(f"Running all 3 methods to find the most reliable prediction...")
    
    image = Image.open(image_path).convert("RGB")
    all_results = []
    
    # Method 1: Basic (no YOLO, no TTA)
    if verbose:
        print(f"\n📊 Method 1: Basic Prediction (no augmentation)...")
    try:
        probs_basic = predict_single(image)
        top_probs, top_indices = torch.topk(probs_basic, min(3, num_classes))
        
        result_basic = {
            'method': 'Basic',
            'predicted_class': class_names[top_indices[0].item()],
            'confidence': top_probs[0].item() * 100,
            'top_k_predictions': [
                {
                    'class': class_names[top_indices[i].item()],
                    'confidence': top_probs[i].item() * 100
                }
                for i in range(len(top_indices))
            ]
        }
        all_results.append(result_basic)
        if verbose:
            print(f"   ✓ {result_basic['predicted_class']} ({result_basic['confidence']:.2f}%)")
    except Exception as e:
        if verbose:
            print(f"   ✗ Failed: {str(e)}")
    
    # Method 2: TTA only (no YOLO)
    if verbose:
        print(f"\n🔬 Method 2: Test-Time Augmentation...")
    try:
        probs_tta = predict_with_tta(image, num_tta=TTA_STEPS)
        top_probs, top_indices = torch.topk(probs_tta, min(3, num_classes))
        
        result_tta = {
            'method': 'TTA',
            'predicted_class': class_names[top_indices[0].item()],
            'confidence': top_probs[0].item() * 100,
            'top_k_predictions': [
                {
                    'class': class_names[top_indices[i].item()],
                    'confidence': top_probs[i].item() * 100
                }
                for i in range(len(top_indices))
            ]
        }
        all_results.append(result_tta)
        if verbose:
            print(f"   ✓ {result_tta['predicted_class']} ({result_tta['confidence']:.2f}%)")
    except Exception as e:
        if verbose:
            print(f"   ✗ Failed: {str(e)}")
    
    # Method 3: YOLO + TTA
    if yolo_model is not None:
        if verbose:
            print(f"\n🎯 Method 3: YOLO Region Detection + TTA...")
        try:
            detected_regions, _ = detect_leaf_regions(image_path)
            
            if detected_regions:
                # Use largest detected region
                region = detected_regions[0]
                probs_yolo_tta = predict_with_tta(region['crop'], num_tta=TTA_STEPS)
                top_probs, top_indices = torch.topk(probs_yolo_tta, min(3, num_classes))
                
                result_yolo_tta = {
                    'method': 'YOLO+TTA',
                    'predicted_class': class_names[top_indices[0].item()],
                    'confidence': top_probs[0].item() * 100,
                    'yolo_region_conf': region['confidence'],
                    'top_k_predictions': [
                        {
                            'class': class_names[top_indices[i].item()],
                            'confidence': top_probs[i].item() * 100
                        }
                        for i in range(len(top_indices))
                    ]
                }
                all_results.append(result_yolo_tta)
                if verbose:
                    print(f"   ✓ {result_yolo_tta['predicted_class']} ({result_yolo_tta['confidence']:.2f}%)")
                    print(f"   Region detected with {region['confidence']:.2f} confidence")
            else:
                if verbose:
                    print(f"   ⚠️  No regions detected, skipping this method")
        except Exception as e:
            if verbose:
                print(f"   ✗ Failed: {str(e)}")
    
    # Analyze results
    if verbose:
        print(f"\n{'='*70}")
        print(f"🔍 ENSEMBLE ANALYSIS")
        print(f"{'='*70}")
    
    if not all_results:
        print("❌ All methods failed!")
        return None
    
    # Check for consensus
    predictions = [r['predicted_class'] for r in all_results]
    prediction_counts = Counter(predictions)
    most_common_pred, count = prediction_counts.most_common(1)[0]
    
    if verbose:
        print(f"\nPredictions by method:")
        for result in all_results:
            print(f"  {result['method']:15} → {result['predicted_class']:35} ({result['confidence']:.2f}%)")
    
    # Calculate scores for each result
    scored_results = []
    for result in all_results:
        score = calculate_prediction_score(result)
        result['reliability_score'] = score
        scored_results.append(result)
    
    # Sort by reliability score
    scored_results.sort(key=lambda x: x['reliability_score'], reverse=True)
    
    if verbose:
        print(f"\nReliability scores:")
        for result in scored_results:
            print(f"  {result['method']:15} → Score: {result['reliability_score']:.2f}")
    
    # Decision logic
    if count == len(all_results):
        # All methods agree
        best_result = max(all_results, key=lambda x: x['confidence'])
        decision_reason = f"✅ UNANIMOUS: All {len(all_results)} methods agree"
        consensus_level = "UNANIMOUS"
    elif count >= 2:
        # Majority vote (2 out of 3)
        majority_results = [r for r in all_results if r['predicted_class'] == most_common_pred]
        best_result = max(majority_results, key=lambda x: x['confidence'])
        decision_reason = f"✅ MAJORITY: {count}/{len(all_results)} methods agree"
        consensus_level = "MAJORITY"
    else:
        # All disagree - use highest reliability score
        best_result = scored_results[0]
        decision_reason = f"⚖️  SPLIT DECISION: Selected based on highest reliability score"
        consensus_level = "SPLIT"
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🏆 FINAL DECISION")
        print(f"{'='*70}")
        print(f"\n{decision_reason}")
        print(f"Selected Method: {best_result['method']}")
        print(f"Prediction: {best_result['predicted_class']}")
        print(f"Confidence: {best_result['confidence']:.2f}%")
        print(f"Reliability Score: {best_result['reliability_score']:.2f}")
        
        # Confidence interpretation
        print(f"\n💡 Assessment:")
        if consensus_level == "UNANIMOUS":
            print(f"   🟢 Very high reliability - all methods agree")
        elif consensus_level == "MAJORITY":
            print(f"   🟡 High reliability - majority consensus")
        else:
            print(f"   🟠 Moderate reliability - methods disagree, using scoring")
        
        if best_result['confidence'] > 90:
            print(f"   🟢 Very high confidence prediction")
        elif best_result['confidence'] > 70:
            print(f"   🟢 High confidence prediction")
        elif best_result['confidence'] > 50:
            print(f"   🟡 Moderate confidence - consider verification")
        else:
            print(f"   🟠 Low confidence - expert verification recommended")
        
        print(f"{'='*70}\n")
    
    # Prepare final result
    final_result = {
        'image_path': image_path,
        'final_prediction': best_result['predicted_class'],
        'final_confidence': best_result['confidence'],
        'selected_method': best_result['method'],
        'reliability_score': best_result['reliability_score'],
        'consensus_level': consensus_level,
        'decision_reason': decision_reason,
        'all_method_results': all_results,
        'top_k_predictions': best_result['top_k_predictions']
    }
    
    return final_result

# -----------------------------
# 1️⃣3️⃣ Visualize Detections
# -----------------------------
def visualize_detections(original_image, detected_regions, predictions, save_path=None):
    """
    Draw bounding boxes and predictions on the original image
    """
    img_draw = original_image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = font
    
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    for idx, (region, pred) in enumerate(zip(detected_regions, predictions)):
        bbox = region['bbox']
        color = colors[idx % len(colors)]
        
        draw.rectangle(bbox, outline=color, width=3)
        
        label = f"{pred['predicted_class']}"
        conf_text = f"{pred['confidence']:.1f}%"
        
        bbox_obj = draw.textbbox((bbox[0], bbox[1]), label, font=font_small)
        text_width = bbox_obj[2] - bbox_obj[0]
        text_height = bbox_obj[3] - bbox_obj[1]
        
        draw.rectangle(
            [bbox[0], bbox[1] - text_height - 5, bbox[0] + text_width + 10, bbox[1]],
            fill=color
        )
        
        draw.text((bbox[0] + 5, bbox[1] - text_height - 5), label, fill=(255, 255, 255), font=font_small)
        draw.text((bbox[0] + 5, bbox[1] + 5), conf_text, fill=color, font=font_small)
    
    if save_path:
        img_draw.save(save_path)
        print(f"📸 Visualization saved to: {save_path}")
    
    return img_draw

# -----------------------------
# 1️⃣4️⃣ Main Prediction Function
# -----------------------------
def predict_image(image_path, top_k=3, show_image=True, use_ensemble=None):
    """
    Predict plant disease with optional ensemble mode
    """
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found — {image_path}")
        return None

    if use_ensemble is None:
        use_ensemble = USE_ENSEMBLE

    try:
        if use_ensemble:
            # Use ensemble prediction
            result = ensemble_predict(image_path, verbose=True)
            
            if show_image:
                try:
                    Image.open(image_path).show()
                except:
                    print("⚠️  Could not display image")
            
            return result
        else:
            # Use simple single method prediction (YOLO + TTA as default)
            print(f"\n{'='*70}")
            print(f"🖼️  Image: {os.path.basename(image_path)}")
            print(f"{'='*70}")
            
            if yolo_model is not None:
                print(f"\n🎯 Using YOLO + TTA method...")
                detected_regions, original_image = detect_leaf_regions(image_path)
                
                if detected_regions:
                    region = detected_regions[0]
                    probs = predict_with_tta(region['crop'], num_tta=TTA_STEPS)
                else:
                    print(f"⚠️  No regions detected, using full image")
                    image = Image.open(image_path).convert("RGB")
                    probs = predict_with_tta(image, num_tta=TTA_STEPS)
            else:
                print(f"\n🔬 Using TTA method...")
                image = Image.open(image_path).convert("RGB")
                probs = predict_with_tta(image, num_tta=TTA_STEPS)
            
            top_probs, top_indices = torch.topk(probs, min(top_k, num_classes))
            
            print(f"\n🏆 Prediction:")
            predicted_label = class_names[top_indices[0].item()]
            confidence = top_probs[0].item() * 100
            print(f"   {predicted_label}")
            print(f"   Confidence: {confidence:.2f}%\n")
            
            if show_image:
                try:
                    Image.open(image_path).show()
                except:
                    print("⚠️  Could not display image")
            
            return {
                'predicted_class': predicted_label,
                'confidence': confidence,
                'top_k_predictions': [
                    {
                        'class': class_names[top_indices[i].item()],
                        'confidence': top_probs[i].item() * 100
                    }
                    for i in range(len(top_indices))
                ]
            }
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# -----------------------------
# 1️⃣5️⃣ Batch Prediction with Ensemble
# -----------------------------
def predict_batch(image_folder, output_json="predictions.json", use_ensemble=None):
    """
    Predict all images in a folder using ensemble mode
    """
    if not os.path.exists(image_folder):
        print(f"❌ Error: Folder not found — {image_folder}")
        return
    
    if use_ensemble is None:
        use_ensemble = USE_ENSEMBLE
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in os.listdir(image_folder)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    if not image_files:
        print(f"❌ No image files found in {image_folder}")
        return
    
    print(f"📂 Found {len(image_files)} images in {image_folder}")
    print(f"🔄 Processing with {'ENSEMBLE mode' if use_ensemble else 'single method'}...\n")
    
    results = []
    for idx, img_file in enumerate(image_files, 1):
        print(f"\n{'='*70}")
        print(f"Processing {idx}/{len(image_files)}: {img_file}")
        print(f"{'='*70}")
        
        img_path = os.path.join(image_folder, img_file)
        result = predict_image(img_path, show_image=False, use_ensemble=use_ensemble)
        
        if result:
            results.append(result)
    
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Predictions saved to {output_json}")
    
    # Summary
    print(f"\n📊 BATCH SUMMARY:")
    print(f"{'='*70}")
    
    if use_ensemble:
        # Ensemble-specific summary
        consensus_counts = Counter([r.get('consensus_level', 'UNKNOWN') for r in results])
        print(f"\nConsensus levels:")
        for level, count in consensus_counts.most_common():
            print(f"  {level}: {count} images")
        
        # Method selection stats
        method_counts = Counter([r.get('selected_method', 'UNKNOWN') for r in results])
        print(f"\nSelected methods:")
        for method, count in method_counts.most_common():
            print(f"  {method}: {count} images")
    
    # Class distribution
    class_counts = {}
    total_confidence = 0
    
    for r in results:
        pred = r.get('final_prediction') or r.get('predicted_class')
        conf = r.get('final_confidence') or r.get('confidence')
        if pred:
            class_counts[pred] = class_counts.get(pred, 0) + 1
            if conf:
                total_confidence += conf
    
    print(f"\nPredicted classes:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count} image(s)")
    
    if results:
        avg_conf = total_confidence / len(results)
        print(f"\n📈 Average confidence: {avg_conf:.2f}%")

# -----------------------------
# 1️⃣6️⃣ Interactive Mode
# -----------------------------
def interactive_mode():
    """Interactive prediction mode"""
    print(f"\n{'='*70}")
    print("🌱 PLANT DISEASE DETECTION - INTELLIGENT ENSEMBLE MODE")
    print(f"{'='*70}\n")
    
    while True:
        print("\nOptions:")
        print("  1. Predict with ENSEMBLE (RECOMMENDED - Runs all 3 methods)")
        print("  2. Predict with single method (YOLO + TTA)")
        print("  3. Batch process folder with ENSEMBLE")
        print("  4. Batch process folder with single method")
        print("  5. Exit")
        
        choice = input("\n📋 Enter your choice (1-5): ").strip()
        
        if choice == '1':
            image_path = input("\n📂 Enter image path: ").strip().strip('"').strip("'")
            predict_image(image_path, use_ensemble=True)
            
        elif choice == '2':
            image_path = input("\n📂 Enter image path: ").strip().strip('"').strip("'")
            predict_image(image_path, use_ensemble=False)
            
        elif choice == '3':
            folder_path = input("\n📂 Enter folder path: ").strip().strip('"').strip("'")
            output_file = input("💾 Output JSON file (default: predictions.json): ").strip()
            if not output_file:
                output_file = "predictions.json"
            predict_batch(folder_path, output_file, use_ensemble=True)
            
        elif choice == '4':
            folder_path = input("\n📂 Enter folder path: ").strip().strip('"').strip("'")
            output_file = input("💾 Output JSON file (default: predictions.json): ").strip()
            if not output_file:
                output_file = "predictions.json"
            predict_batch(folder_path, output_file, use_ensemble=False)
            
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

# -----------------------------
# 1️⃣7️⃣ Main Entry Point
# -----------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Parse command line arguments
        use_ensemble_arg = '--no-ensemble' not in sys.argv
        
        predict_image(image_path, use_ensemble=use_ensemble_arg)
    else:
        interactive_mode()