import os
import random
import numpy as np
import json
import torch
import h5py
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocess import audio_to_spectrogram
from dataset import ASVspoofDataset
from model import MultiModalAcousticNet

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
        print(f"Starting preprocessing for: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)

        files = [f for f in os.listdir(input_dir) if f.endswith('.flac')]
        for i, file in enumerate(files):
            audio_path = os.path.join(input_dir, file)

            audio_to_spectrogram(audio_path, output_dir=output_dir)

            if (i + 1) % 500 == 0:
                print(f"  -> Processed {i + 1}/{len(files)} files...")
    else:
        print(f"Skipping preprocessing: Data already exists in {output_dir}")


def train_model():
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 20

    train_protocol = r"D:\InfoTect\model\data\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
    val_protocol = r"D:\InfoTect\model\data\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"

    train_raw_audio = r"D:\InfoTect\model\data\ASVspoof2019_LA_train\flac"
    val_raw_audio = r"D:\InfoTect\model\data\ASVspoof2019_LA_dev\flac"

    train_preprocessed_dir = r"D:\InfoTect\model\preprocessed_train_acoustic_reports"
    val_preprocessed_dir = r"D:\InfoTect\model\preprocessed_val_acoustic_reports"

    print("\n--- STEP 1: Running Preprocessing Pipeline ---")
    process_directory(train_raw_audio, train_preprocessed_dir)
    process_directory(val_raw_audio, val_preprocessed_dir)

    print("\n--- STEP 2: Initializing MultiModalAcousticNet Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_dataset = ASVspoofDataset(protocol_path=train_protocol, preprocessed_dir=train_preprocessed_dir)
    
    acoustic_stats = train_dataset.get_acoustic_stats()
    
    save_dir = "/kaggle/working/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    stats_out_path = os.path.join(save_dir, "acoustic_stats.json")
    
    with open(stats_out_path, "w") as f:
        json.dump({
            "acoustic_mean": acoustic_stats[0].tolist(),
            "acoustic_std": acoustic_stats[1].tolist()
        }, f, indent=4)
        
    print(f"Saved acoustic normalization stats to {stats_out_path}")
    
    val_dataset = ASVspoofDataset(
        protocol_path=val_protocol,
        preprocessed_dir=val_preprocessed_dir,
        acoustic_stats=acoustic_stats,
    )

    train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=min(8, os.cpu_count()),
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model = MultiModalAcousticNet().to(device)
    
    model = model.to(memory_format=torch.channels_last)

        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, threshold=1e-4, min_lr=1e-7)
    
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_loss = float('inf')
    start_epoch = 0

    checkpoint_path = os.path.join(save_dir, "last_checkpoint.pt")

    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]

        print(f"Resuming training from Epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting from scratch.")

    if torch.__version__.startswith("2"):
        model = torch.compile(model)
        
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for spec, room, acoustic, labels in train_loader:
            spec = spec.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last
            )
            room = room.to(device, non_blocking=True)
            acoustic = acoustic.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(spec, room, acoustic)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * spec.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        if total_train > 0:
            epoch_train_loss = running_loss / total_train
            epoch_train_acc = (correct_train / total_train) * 100
        else:
            print(f"\n[ERROR] total_train is 0! running_loss accumulated: {running_loss}")
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for spec, room, acoustic, labels in val_loader:

                spec = spec.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last
                )
                room = room.to(device, non_blocking=True)
                acoustic = acoustic.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(
                    "cuda",
                    enabled=torch.cuda.is_available()
                ):
                    outputs = model(spec, room, acoustic)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * spec.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100
        scheduler.step(epoch_val_loss)
        
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current Learning Rate: {current_lr:.8f}")

        is_best = False

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            is_best = True
            
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'learning_rate': current_lr,
            'best_val_loss': best_val_loss,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
        }
        
        if is_best:
            best_model_path = os.path.join(save_dir, "best_multimodal_acoustic_net.pth")
            torch.save(model.state_dict(),best_model_path)
            print(f">>> Validation loss improved! Saved best weights to {best_model_path}")


        checkpoint_path = os.path.join(save_dir, "last_checkpoint.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved -> {checkpoint_path}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path_5 = os.path.join(
                save_dir,
                f"checkpoint_epoch_{epoch+1}.pt"
            )
            torch.save(checkpoint, checkpoint_path_5)
            print(f"Checkpoint saved -> {checkpoint_path_5}")


    final_model_h5_path = os.path.join(save_dir, "final_multimodal_acoustic_net.h5")

    
    final_model_path = os.path.join(
        save_dir,
        "final_multimodal_acoustic_net.pth"
    )

    torch.save(
        model.state_dict(),
        final_model_path
    )

    print(
        f"\nTraining completed! Saved final model to {final_model_path}"
    )


if __name__ == "__main__":
    train_model()
