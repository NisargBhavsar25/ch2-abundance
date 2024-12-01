import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

class MoonDataset(Dataset):
    def __init__(self, patches_dir, transform=None):
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor()
        ])
        
        print(f"Looking for images in directory: {patches_dir}")
        patches_dir_abs = os.path.abspath(patches_dir)
        print(f"Absolute path: {patches_dir_abs}")
        print(f"Directory exists: {os.path.exists(patches_dir_abs)}")
        
        self.low_patches = glob(os.path.join(patches_dir, 'patch_Fe_151x151_xy(*).png'))
        print(f"Found {len(self.low_patches)} Low resolution patches")
        
        self.pairs = []
        
        for low_path in self.low_patches:
            coords = Path(low_path).stem.split('_')[-1]
            req_path = os.path.join(patches_dir, f'patch_Req_15x15_{coords}.png')
            
            if os.path.exists(req_path):
                self.pairs.append((low_path, req_path))
            else:
                print(f"Missing Req patch for coords: {coords}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, req_path = self.pairs[idx]
        
        # Load images
        low_img = Image.open(low_path).convert('RGB')
        req_img = Image.open(req_path).convert('RGB')
        
        # Apply transforms
        low_tensor = self.transform(low_img)
        req_tensor = self.transform(req_img)
        
        return low_tensor, req_tensor

# Create datasets
dataset = MoonDataset('fe_patches')
print(f"Number of image pairs found: {len(dataset)}")

# Add a guard clause to prevent DataLoader creation if dataset is empty
if len(dataset) == 0:
    raise ValueError("Dataset is empty! Please check if the patches directory exists and contains images.")

# Create data loaders
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
