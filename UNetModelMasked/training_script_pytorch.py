
# Standard libraries
import os
import glob
import argparse
import logging
import sys
import json
import numpy as np

# PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================
# PyTorch 3D U-Net Model
# ============================
class UNet3D(nn.Module):
    """
    3D U-Net implementation for inpainting tasks.
    - in_channels: number of input channels (1 for observed only, 2 if mask is used)
    - out_channels: number of output channels (usually 1)
    - base_filters: number of filters in the first layer
    - depth: number of downsampling/upsampling steps
    - output_activation: activation function for output layer
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=16, depth=4, output_activation='softplus'):
        super(UNet3D, self).__init__()
        self.depth = depth
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        filters = base_filters
        # Encoder
        for d in range(depth):
            self.down_convs.append(nn.Sequential(
                nn.Conv3d(in_channels if d==0 else filters, filters, 3, padding=1),
                nn.ReLU(),
                nn.Conv3d(filters, filters, 3, padding=1),
                nn.ReLU()
            ))
            in_channels = filters
            filters *= 2
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(filters//2, filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(filters, filters, 3, padding=1),
            nn.ReLU()
        )
        # Decoder
        for d in reversed(range(depth)):
            filters //= 2
            self.up_convs.append(nn.Sequential(
                nn.ConvTranspose3d(filters*2, filters, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.Conv3d(filters*2, filters, 3, padding=1),
                nn.ReLU(),
                nn.Conv3d(filters, filters, 3, padding=1),
                nn.ReLU()
            ))
        self.final = nn.Conv3d(base_filters, out_channels, 3, padding=1)
        self.output_activation = output_activation
    def forward(self, x):
        # Encoder path: save outputs for skip connections
        encs = []
        for down in self.down_convs:
            x = down(x)
            encs.append(x)
            x = nn.MaxPool3d(2)(x)
        # Bottleneck
        x = self.bottleneck(x)
        # Decoder path: upsample and concatenate skip connections
        for i, up in enumerate(self.up_convs):
            x = up(x)
            x = torch.cat([x, encs[-(i+1)]], dim=1)
        # Final output layer
        x = self.final(x)
        if self.output_activation == 'softplus':
            x = nn.functional.softplus(x)
        return x

# ============================
# PyTorch Dataset
# ============================
class NpyDataset(Dataset):
    """
    PyTorch Dataset for loading .npy files for observed, true, and (optionally) mask data.
    Normalizes data and prepares input for UNet3D.
    """
    def __init__(self, obs_files, true_files, mask_files=None, use_mask=False, normalization=40.0):
        self.obs_files = obs_files
        self.true_files = true_files
        self.mask_files = mask_files
        self.use_mask = use_mask
        self.normalization = normalization
    def __len__(self):
        return len(self.obs_files)
    def __getitem__(self, idx):
        # Load and normalize observed and true fields
        obs = np.load(self.obs_files[idx]) / self.normalization
        true = np.load(self.true_files[idx]) / self.normalization
        obs = np.expand_dims(obs, axis=0) # (1, D, H, W)
        true = np.expand_dims(true, axis=0)
        if self.use_mask and self.mask_files is not None:
            # Load mask and concatenate as second channel
            mask = np.load(self.mask_files[idx])
            mask = np.expand_dims(mask, axis=0)
            x = np.concatenate([obs, mask], axis=0)
        else:
            x = obs
        # Return input and target as tensors
        return torch.tensor(x, dtype=torch.float32), torch.tensor(true, dtype=torch.float32)

# ============================
# Training script
# ============================
def main():
    """
    Main training loop for PyTorch UNet3D inpainting.
    Loads data, trains model, saves losses and predictions.
    """
    parser = argparse.ArgumentParser(description='Train 3D U-Net for inpainting (PyTorch).')
    parser.add_argument('--param_file', type=str, help='Path to JSON parameter file')
    parser.add_argument('--obs_dir', type=str, required=True)
    parser.add_argument('--true_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str)
    parser.add_argument('--use_mask', action='store_true')
    parser.add_argument('--field_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='output_products')
    parser.add_argument('--log_file', type=str, default='logs/training.log')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(args.log_file), logging.StreamHandler(sys.stdout)])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Collect file lists
    obs_files = sorted(glob.glob(os.path.join(args.obs_dir, '*.npy')))
    true_files = sorted(glob.glob(os.path.join(args.true_dir, '*.npy')))
    mask_files = sorted(glob.glob(os.path.join(args.mask_dir, '*.npy'))) if args.mask_dir else None
    assert len(obs_files) == len(true_files), "Mismatch in number of observed and true files!"
    if args.use_mask and mask_files:
        assert len(mask_files) == len(obs_files), "Mismatch in number of mask files!"
    # Split train/validation
    split = int(0.8 * len(obs_files))
    train_obs, val_obs = obs_files[:split], obs_files[split:]
    train_true, val_true = true_files[:split], true_files[split:]
    if args.use_mask and mask_files:
        train_mask, val_mask = mask_files[:split], mask_files[split:]
    else:
        train_mask = val_mask = None

    # Create PyTorch datasets and loaders
    train_dataset = NpyDataset(train_obs, train_true, train_mask, args.use_mask)
    val_dataset = NpyDataset(val_obs, val_true, val_mask, args.use_mask)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Build model, optimizer, and loss
    in_channels = 2 if args.use_mask else 1
    model = UNet3D(in_channels=in_channels, out_channels=1, base_filters=16, depth=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('store_models', exist_ok=True)
    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_running_loss += loss.item() * x.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        logging.info(f'Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}')
        # Save model every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'store_models/model_{epoch:02d}.pt')
    # Save loss curves
    np.save('losses/loss.npy', np.array(train_losses))
    np.save('losses/val_loss.npy', np.array(val_losses))
    # Save predictions on validation set
    model.eval()
    for i, (x, y) in enumerate(val_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()
        for j in range(pred.shape[0]):
            # Denormalize and save predicted field
            np.save(os.path.join(args.output_dir, f'pred_field_{i*args.batch_size+j:03d}.npy'), pred[j,0] * 40.0)

if __name__ == '__main__':
    main()
