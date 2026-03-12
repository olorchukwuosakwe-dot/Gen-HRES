# train_cgan.py
# Conditional GAN for extreme weather synthesis - ready for Milestone 1

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# ====================== CONFIG ======================
device = torch.device("cpu")  # your Mac will run fine
batch_size = 64
lr = 0.0002
epochs = 200          # increase to 500 later if you want
noise_dim = 100
num_conditions = 2    # season (0/1) + extreme (0/1)

# Load your preprocessed files (from the script you just ran)
profiles = np.load('preprocessed_daily_profiles.npy')           # (days, 24, 3)
labels_df = pd.read_csv('daily_labels.csv')

# Convert labels to one-hot style
labels = np.zeros((len(labels_df), num_conditions))
labels[:, 0] = (labels_df['season'] == 'Wet').astype(float)
labels[:, 1] = labels_df['is_extreme'].astype(float)

print(f"Loaded {profiles.shape[0]} daily profiles | Extreme days: {labels[:,1].sum():.0f}")

# ====================== DATASET ======================
class WeatherDataset(Dataset):
    def __init__(self, profiles, labels):
        self.profiles = torch.tensor(profiles, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.profiles)
    
    def __getitem__(self, idx):
        return self.profiles[idx], self.labels[idx]

dataset = WeatherDataset(profiles, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ====================== MODELS ======================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Linear(num_conditions, 24*3)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + 24*3, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 24*3),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_emb = self.label_embed(labels).view(labels.size(0), -1)
        x = torch.cat([noise, label_emb], dim=1)
        return self.model(x).view(-1, 24, 3)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Linear(num_conditions, 24*3)
        self.model = nn.Sequential(
            nn.Linear(24*3 + 24*3, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, profiles, labels):
        label_emb = self.label_embed(labels).view(labels.size(0), -1)
        x = torch.cat([profiles.view(profiles.size(0), -1), label_emb], dim=1)
        return self.model(x)

G = Generator().to(device)
D = Discriminator().to(device)

opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# ====================== TRAINING ======================
print("Starting cGAN training...")
for epoch in range(epochs):
    for real_profiles, cond in dataloader:
        real_profiles = real_profiles.to(device)
        cond = cond.to(device)
        batch_size = real_profiles.size(0)
        
        # Train Discriminator
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_profiles = G(noise, cond)
        
        real_pred = D(real_profiles, cond)
        fake_pred = D(fake_profiles.detach(), cond)
        
        d_loss = criterion(real_pred, real_labels) + criterion(fake_pred, fake_labels)
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()
        
        # Train Generator
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_profiles = G(noise, cond)
        fake_pred = D(fake_profiles, cond)
        g_loss = criterion(fake_pred, real_labels)
        
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

# ====================== SAVE & GENERATE 1000 EXTREME PROFILES ======================
torch.save(G.state_dict(), 'cgan_generator.pth')
print("✅ Model saved: cgan_generator.pth")

# 1. Generate 1000 extreme profiles in normalized space [-1, 1]
G.eval()
with torch.no_grad():
    extreme_cond = torch.tensor([[0, 1.0]] * 1000).to(device)  # force "extreme"
    noise = torch.randn(1000, noise_dim).to(device)
    generated_norm = G(noise, extreme_cond).cpu().numpy() # Shape: (1000, 24, 3)

# 2. Extract original Min/Max values for Denormalization
print("Reversing normalization to real-world physics...")
orig_df = pd.read_csv('preprocessed_long.csv')
features = ["ALLSKY_SFC_SW_DWN", "WS50M", "T2M"]
mins = orig_df[features].min().values
maxs = orig_df[features].max().values

# 3. Denormalize: Reverse the formula from the preprocessing script
# Original formula: norm = 2 * ((val - min) / (max - min)) - 1
# Reversed formula: val = ((norm + 1) / 2) * (max - min) + min
generated_real = ((generated_norm + 1) / 2) * (maxs - mins) + mins

# Prevent tiny negative hardware glitches (e.g., -0.0001 irradiance due to GAN noise)
generated_real = np.clip(generated_real, a_min=0.0, a_max=None)

# 4. Flatten the 1000 days into a continuous 24,000-hour timeline for Simulink
flat_data = generated_real.reshape(-1, 3)
out_df = pd.DataFrame(flat_data, columns=['Solar_Irradiance_W_m2', 'Wind_Speed_m_s', 'Temperature_C'])

# 5. Export for MATLAB
out_df.to_csv('simulink_extreme_forecast.csv', index=False)
print("✅ Generated 24,000 continuous hours of extreme weather → simulink_extreme_forecast.csv")
print("Ready for Milestone 1 (import into MATLAB/Simulink as disturbance forecast)!")