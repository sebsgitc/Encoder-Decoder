import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from prepare_dataset import LungSegmentationDataset, prepare_data, MODEL_DIR
from unet_model import UNet
from tqdm import tqdm

NUM_EPOCHS = 5
LR = 0.05
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 2

def train_model():
    # Prepare data
    train_images, val_images, train_vessels, val_vessels = prepare_data()
    
    # Create datasets
    train_dataset = LungSegmentationDataset(train_images, train_vessels)
    val_dataset = LungSegmentationDataset(val_images, val_vessels)
    
    # Create dataloaders with smaller batch size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,  # Reduced for memory efficiency
        shuffle=True,
        num_workers=0  # No multiprocessing
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = UNet(n_channels=1, n_classes=3).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Training loop
    num_epochs = NUM_EPOCHS
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Clear cache at start of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': epoch_loss/len(train_loader)})
                
                # Clear memory after each batch
                del images, masks, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, masks).item()
            print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
    
    # Save model with timestamp
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(MODEL_DIR, f'lung_segmentation_model_{timestamp}.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()