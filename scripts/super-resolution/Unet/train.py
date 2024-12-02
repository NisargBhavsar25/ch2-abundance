import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset import train_loader
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = deeplabv3_resnet50(pretrained=True)
model.classifier[-1] = nn.Conv2d(256, 3, kernel_size=1)  # Change output to 3 channels
model = model.to(device)

# Initialize loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# Training loop
num_epochs = 10
best_loss = float('inf')
save_dir = 'checkpoints'
os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Progress bar for batches
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for i, (fe_images, req_images) in enumerate(pbar):
        # Clear cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        fe_images = fe_images.to(device)
        req_images = req_images.to(device)
        
        # Forward pass
        outputs = model(fe_images)['out']  # DeepLabV3 returns a dict with 'out' key
        loss = criterion(outputs, req_images)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update running loss and progress bar before cleanup
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
        
        # Clear variables to free memory
        del outputs
        del loss
        torch.cuda.empty_cache()  # Clear cache again after operations
    
    # Step the scheduler
    scheduler.step()
    
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
    
    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': best_loss,
        }, os.path.join(save_dir, 'best_model.pth'))

print('Training finished!')
