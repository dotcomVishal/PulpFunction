import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import os
import json
from collections import Counter
from datetime import datetime
import platform

# =========================================================
# CONFIGURATION
# =========================================================
DATA_DIR = "data"
MODEL_PATH = "plant_disease_model.pth"
BEST_MODEL_PATH = "plant_disease_best.pth"
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "training_log.json"

BATCH_SIZE = 32  # Fixed: Increased from 16
EPOCHS = 20
INITIAL_LR = 3e-4  # Fixed: Changed from 1e-5 (too small)
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed: Platform-specific workers (Windows compatibility)
if platform.system() == 'Windows':
    NUM_WORKERS = 0
    PERSISTENT_WORKERS = False
else:
    NUM_WORKERS = 4
    PERSISTENT_WORKERS = True

EARLY_STOP_PATIENCE = 7  # Added: Early stopping
MIXED_PRECISION = True  # Added: Faster training

# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def calculate_metrics(outputs, labels):
    """Calculate accuracy and predictions"""
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct, preds

def evaluate(model, loader, criterion, scaler, desc="Evaluating"):
    """Evaluation function"""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            if MIXED_PRECISION and scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            correct, preds = calculate_metrics(outputs, labels)
            total_correct += correct
            total_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    accuracy = 100 * total_correct / total_samples
    
    return avg_loss, accuracy, all_preds, all_labels

def save_checkpoint(epoch, model, optimizer, scheduler, val_acc, train_dataset, is_best=False, training_history=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'classes': train_dataset.classes,
        'training_history': training_history,
        'timestamp': datetime.now().isoformat()
    }
    
    if is_best:
        torch.save(checkpoint, BEST_MODEL_PATH)
        print(f"💾 Best model saved: {BEST_MODEL_PATH}")
    
    # Periodic checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

def load_checkpoint(path):
    """Load checkpoint for resuming"""
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location=DEVICE)
    return checkpoint

def save_training_log(history):
    """Save training history to JSON"""
    with open(LOG_FILE, 'w') as f:
        json.dump(history, f, indent=2)

# =========================================================
# MAIN TRAINING FUNCTION
# =========================================================
def main():
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"🌱 PLANT DISEASE CLASSIFICATION TRAINING")
    print(f"{'='*70}")
    print(f"📱 Device: {DEVICE}")
    print(f"💻 Platform: {platform.system()}")
    print(f"🔢 Batch Size: {BATCH_SIZE}")
    print(f"📊 Epochs: {EPOCHS}")
    print(f"📚 Learning Rate: {INITIAL_LR}")
    print(f"👷 Workers: {NUM_WORKERS}")
    print(f"⚡ Mixed Precision: {MIXED_PRECISION}")
    print(f"{'='*70}\n")

    # =========================================================
    # DATA TRANSFORMS - Fixed: Better augmentation and test transform
    # =========================================================
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # More aggressive
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Added
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Added
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))  # Added: Cutout augmentation
    ])

    # Fixed: Proper test transform (Resize → CenterCrop instead of direct resize)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # =========================================================
    # LOAD DATASETS
    # =========================================================
    print("📂 Loading datasets...")
    train_dataset = datasets.ImageFolder(root=f"{DATA_DIR}/train", transform=train_transform)
    test_dataset = datasets.ImageFolder(root=f"{DATA_DIR}/test", transform=test_transform)

    num_classes = len(train_dataset.classes)
    print(f"\n✅ Datasets loaded successfully!")
    print(f"📋 Classes: {num_classes}")
    print(f"📊 Total Train Images: {len(train_dataset)}")
    print(f"📊 Total Test Images: {len(test_dataset)}")
    print(f"\n🏷️  Class names: {', '.join(train_dataset.classes)}")

    # Added: Analyze class distribution for imbalance
    class_counts = Counter([label for _, label in train_dataset.samples])
    print(f"\n📈 Class Distribution:")
    print(f"{'Class':<40} {'Count':>8} {'%':>8}")
    print(f"{'-'*58}")
    for idx in range(num_classes):
        count = class_counts[idx]
        percentage = 100 * count / len(train_dataset)
        print(f"{train_dataset.classes[idx]:<40} {count:>8} {percentage:>7.1f}%")

    # Calculate imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"\n⚖️  Imbalance Ratio: {imbalance_ratio:.2f}:1", end="")
    if imbalance_ratio > 10:
        print(" (⚠️  High imbalance - using weighted sampling)")
    elif imbalance_ratio > 3:
        print(" (⚠️  Moderate imbalance - using weighted sampling)")
    else:
        print(" (✅ Balanced dataset)")

    # =========================================================
    # SPLIT TRAIN → TRAIN + VAL
    # =========================================================
    print("\n🔀 Creating train-validation split...")
    
    # First, create a base dataset for validation with val transforms
    val_base_dataset = datasets.ImageFolder(root=f"{DATA_DIR}/train", transform=val_transform)
    
    val_size = int(VAL_SPLIT * len(train_dataset))
    train_size = len(train_dataset) - val_size
    
    # Split indices
    train_ds, val_ds = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create a new Subset for validation with correct transforms
    from torch.utils.data import Subset
    val_ds = Subset(val_base_dataset, val_ds.indices)

    print(f"✅ Train samples: {len(train_ds)}")
    print(f"✅ Validation samples: {len(val_ds)}")

    # =========================================================
    # WEIGHTED SAMPLING - Added: Handle class imbalance
    # =========================================================
    print("\n⚖️  Computing class weights for balanced sampling...")
    class_weights = 1.0 / torch.tensor([class_counts[i] for i in range(num_classes)], dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(DEVICE)

    train_targets = [train_dataset.targets[i] for i in train_ds.indices]
    sample_weights = [class_weights[t].cpu().item() for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # DataLoader setup
    dataloader_kwargs = {
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'pin_memory': torch.cuda.is_available()
    }
    
    if PERSISTENT_WORKERS and NUM_WORKERS > 0:
        dataloader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(train_ds, sampler=sampler, **dataloader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

    # =========================================================
    # MODEL SETUP - Fixed: Better architecture
    # =========================================================
    print("\n🏗️  Building model...")
    
    # Fixed: Use weights instead of pretrained (deprecated)
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    
    # Fixed: Freeze early layers for transfer learning
    for param in model.features[:5].parameters():
        param.requires_grad = False

    num_features = model.classifier[1].in_features
    
    # Fixed: Enhanced classifier with dropout and batch norm
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    
    model = model.to(DEVICE)
    
    print(f"✅ Model: EfficientNet-B0 with enhanced classifier")
    print(f"✅ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # =========================================================
    # LOSS, OPTIMIZER, SCHEDULER - Fixed: Better training setup
    # =========================================================
    # Fixed: Add weighted loss and label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Fixed: Use AdamW with weight decay instead of Adam
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': INITIAL_LR * 0.1},
        {'params': model.classifier.parameters(), 'lr': INITIAL_LR}
    ], weight_decay=0.01)
    
    # Added: Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    # Added: Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if MIXED_PRECISION and torch.cuda.is_available() else None

    # =========================================================
    # RESUME TRAINING CHECK - Added: Checkpoint support
    # =========================================================
    best_val_acc = 0.0
    patience_counter = 0
    start_epoch = 0
    training_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rates': []
    }

    if os.path.exists(BEST_MODEL_PATH):
        print(f"\n🔄 Found existing checkpoint: {BEST_MODEL_PATH}")
        resume = input("📋 Resume training? (y/n): ").lower().strip()
        if resume == 'y':
            checkpoint = load_checkpoint(BEST_MODEL_PATH)
            if checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_acc = checkpoint['val_acc']
                if 'training_history' in checkpoint and checkpoint['training_history']:
                    training_history = checkpoint['training_history']
                print(f"✅ Resumed from epoch {checkpoint['epoch']} | Val Acc: {best_val_acc:.2f}%")

    # =========================================================
    # TRAINING LOOP - Fixed: Add gradient clipping and better monitoring
    # =========================================================
    print(f"\n{'='*70}")
    print(f"🚀 STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Starting from epoch: {start_epoch + 1}/{EPOCHS}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, EPOCHS):
        # Added: Unfreeze all layers after epoch 5
        if epoch == 5:
            for param in model.features.parameters():
                param.requires_grad = True
            print("🔓 Unfroze all layers for fine-tuning\n")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            
            # Added: Mixed precision training
            if MIXED_PRECISION and scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Added: Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                # Added: Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            batch_correct, _ = calculate_metrics(outputs, labels)
            correct += batch_correct
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
        
        scheduler.step()
        
        # Calculate metrics
        train_loss = running_loss / total
        train_acc = 100 * correct / total
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, scaler, "Validating")
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rates'].append(current_lr)
        
        # Print summary
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {current_lr:.2e} | Best Val: {best_val_acc:.2f}%")
        
        # Added: Save best model
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            save_checkpoint(epoch, model, optimizer, scheduler, val_acc, train_dataset, 
                          is_best=True, training_history=training_history)
            print(f"  ✅ New best! +{improvement:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
        
        # Periodic checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, val_acc, train_dataset, 
                       is_best=False, training_history=training_history)
        
        # Save training log
        save_training_log(training_history)
        print(f"{'-'*70}\n")
        
        # Added: Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"⚠️  Early stopping after {epoch+1} epochs (no improvement for {EARLY_STOP_PATIENCE} epochs)\n")
            break

    # =========================================================
    # TEST EVALUATION - Fixed: Load best model
    # =========================================================
    print(f"\n{'='*70}")
    print(f"🎯 FINAL EVALUATION")
    print(f"{'='*70}\n")

    # Fixed: Load best model instead of using last epoch
    print("📥 Loading best model for testing...")
    checkpoint = load_checkpoint(BEST_MODEL_PATH)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded best model from epoch {checkpoint['epoch']}\n")

    # Test evaluation
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, scaler, "Testing")

    print(f"\n{'='*70}")
    print(f"📈 FINAL RESULTS")
    print(f"{'='*70}")
    print(f"✅ Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"✅ Final Test Accuracy:      {test_acc:.2f}%")
    print(f"✅ Final Test Loss:          {test_loss:.4f}")
    print(f"{'='*70}\n")

    # Added: Per-class accuracy
    print("📊 Per-Class Test Accuracy:")
    print(f"{'Class':<40} {'Accuracy':>10}")
    print(f"{'-'*52}")
    import numpy as np
    test_preds_np = np.array(test_preds)
    test_labels_np = np.array(test_labels)
    for idx, class_name in enumerate(train_dataset.classes):
        class_mask = test_labels_np == idx
        if class_mask.sum() > 0:
            class_acc = 100 * (test_preds_np[class_mask] == test_labels_np[class_mask]).sum() / class_mask.sum()
            print(f"{class_name:<40} {class_acc:>9.2f}%")

    # =========================================================
    # SAVE MODEL - Fixed: Save with metadata
    # =========================================================
    print(f"\n{'='*70}")
    print("💾 Saving final model...")
    
    # Fixed: Save complete model info
    final_save_dict = {
        'model_state_dict': model.state_dict(),
        'classes': train_dataset.classes,
        'test_acc': test_acc,
        'val_acc': best_val_acc,
        'training_history': training_history,
        'num_classes': num_classes,
        'model_architecture': 'efficientnet_b0',
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(final_save_dict, MODEL_PATH)
    print(f"✅ Model saved: {MODEL_PATH}")
    print(f"✅ Training log: {LOG_FILE}")
    print(f"✅ Checkpoints: {CHECKPOINT_DIR}/")
    print(f"{'='*70}\n")
    print("🎉 Training completed successfully!")


# =========================================================
# ENTRY POINT - Fixed: Windows multiprocessing support
# =========================================================
if __name__ == '__main__':
    main()