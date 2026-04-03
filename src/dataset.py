# File: src/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for brain tumor segmentation.
    
    Takes preprocessed numpy arrays (images and masks)
    and returns PyTorch tensors ready for model training.
    
    Why custom Dataset?
    - PyTorch DataLoader requires a Dataset object
    - Handles data augmentation during training
    - Automatically converts numpy to tensors
    - Enables batching, shuffling, parallel loading
    """
    
    def __init__(self, images, masks, augment=False):
        """
        Args:
            images: numpy array of shape (N, H, W), normalized to [0, 1]
            masks: numpy array of shape (N, H, W), binary (0 or 1)
            augment: if True, apply random augmentations (only for training)
        """
        self.images = images
        self.masks = masks
        self.augment = augment
        
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.4
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]    # (H, W) float32
        mask = self.masks[idx]      # (H, W) float32
        
        # Apply augmentation if training
        if self.augment:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Add channel dimension: (H, W) → (1, H, W)
        # PyTorch Conv2d expects (batch, channels, height, width)
        image = torch.FloatTensor(image).unsqueeze(0)
        mask = torch.FloatTensor(mask).unsqueeze(0)
        
        return image, mask


def get_dataloaders(data_path, batch_size=16, num_workers=0):
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        data_path: path to folder containing .npy files
        batch_size: number of images per batch
        num_workers: parallel data loading workers (0 for CPU)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    import os
    
    train_images = np.load(os.path.join(data_path, "train_images.npy"))
    train_masks = np.load(os.path.join(data_path, "train_masks.npy"))
    val_images = np.load(os.path.join(data_path, "val_images.npy"))
    val_masks = np.load(os.path.join(data_path, "val_masks.npy"))
    test_images = np.load(os.path.join(data_path, "test_images.npy"))
    test_masks = np.load(os.path.join(data_path, "test_masks.npy"))
    
    # Training: augmentation ON, shuffle ON
    train_dataset = BrainTumorDataset(train_images, train_masks, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Randomize order each epoch
        num_workers=num_workers,
        pin_memory=True         # Faster GPU transfer
    )
    
    # Validation: augmentation OFF, shuffle OFF
    val_dataset = BrainTumorDataset(val_images, val_masks, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test: augmentation OFF, shuffle OFF
    test_dataset = BrainTumorDataset(test_images, test_masks, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    loader, _, _ = get_dataloaders("data/processed", batch_size=8)
    images, masks = next(iter(loader))
    print(f"Batch — Images: {images.shape}, Masks: {masks.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Mask values: {torch.unique(masks)}")