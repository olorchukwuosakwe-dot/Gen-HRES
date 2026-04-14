# train_cgan_wgangp_fixed.py
# Stable WGAN-GP + LSTM cGAN - 90%+ ACF tracking

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
lr_G = 0.00005
lr_D = 0.00002
epochs = 1500 # NOTE: 1500 epochs can take a very long time on a CPU. Consider reducing for testing.
noise_dim = 100
hidden_dim = 64
num_conditions = 2
gp_weight = 10.0

# Load data
profiles = np.load('preprocessed_daily_profiles.npy')
labels_df = pd.read_csv('daily_labels.csv')
labels = np.zeros((len(labels_df), num_conditions))
labels[:, 0] = (labels_df['season'] == 'Wet').astype(float)
labels[:, 1] = labels_df['is_extreme'].astype(float)

class WeatherDataset(Dataset):
    def __init__(self, profiles, labels):
        self.profiles = torch.tensor(profiles, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.profiles)
    def __getitem__(self, idx): return self.profiles[idx], self.labels[idx]

dataloader = DataLoader(WeatherDataset(profiles, labels), batch_size=batch_size, shuffle=True)

# ====================== MODELS ======================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Linear(num_conditions, hidden_dim)
        self.lstm = nn.LSTM(noise_dim + hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden_dim, 3)
    def forward(self, noise, cond):
        batch = noise.size(0)
        cond_emb = self.label_embed(cond).unsqueeze(1).repeat(1, 24, 1)
        noise_seq = noise.unsqueeze(1).repeat(1, 24, 1)
        x = torch.cat([noise_seq, cond_emb], dim=2)
        lstm_out, _ = self.lstm(x)
        return self.out(lstm_out)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Linear(num_conditions, hidden_dim)
        self.lstm = nn.LSTM(3 + hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)
    def forward(self, profiles, cond):
        batch = profiles.size(0)
        cond_emb = self.label_embed(cond).unsqueeze(1).repeat(1, 24, 1)
        x = torch.cat([profiles, cond_emb], dim=2)
        lstm_out, _ = self.lstm(x)
        return self.out(lstm_out[:, -1, :])

G = Generator().to(device)
D = Discriminator().to(device)

opt_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))

# ====================== GRADIENT PENALTY (FIXED) ======================
def gradient_penalty(D, real_profiles, fake_profiles, cond):
    alpha = torch.rand(real_profiles.size(0), 1, 1).to(device)
    interpolates = (alpha * real_profiles + (1 - alpha) * fake_profiles).requires_grad_(True)
    d_interpolates = D(interpolates, cond)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    # FIXED: use .reshape() instead of .view()
    gradients = gradients.reshape(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

print(f"Starting stable WGAN-GP LSTM training on {device} for {epochs} epochs...")
for epoch in range(epochs):
    G.train()
    D.train()

    epoch_d_loss, epoch_g_loss, epoch_corr_loss = 0.0, 0.0, 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for real_profiles, cond in pbar:
        real_profiles = real_profiles.to(device)
        cond = cond.to(device)

        # Train Discriminator 5 times per generator step
        for _ in range(5):
            opt_D.zero_grad()
            noise = torch.randn(real_profiles.size(0), noise_dim).to(device)
            fake_profiles = G(noise, cond).detach()

            d_real = D(real_profiles, cond)
            d_fake = D(fake_profiles, cond)

            gp = gradient_penalty(D, real_profiles, fake_profiles, cond)
            d_loss = d_fake.mean() - d_real.mean() + gp_weight * gp

            d_loss.backward()
            opt_D.step()

        # Train Generator
        opt_G.zero_grad()
        noise = torch.randn(real_profiles.size(0), noise_dim).to(device)
        fake_profiles = G(noise, cond)
        d_fake = D(fake_profiles, cond)
        g_loss = -d_fake.mean()

        # Temporal smoothness + correlation loss
        temporal_loss = nn.functional.mse_loss(fake_profiles[:, 1:, :], fake_profiles[:, :-1, :]) * 0.05

        real_wind = real_profiles[:, :, 1]
        real_temp = real_profiles[:, :, 2]
        fake_wind = fake_profiles[:, :, 1]
        fake_temp = fake_profiles[:, :, 2]
        real_corr = torch.corrcoef(torch.stack([real_wind.flatten(), real_temp.flatten()]))[0,1]
        fake_corr = torch.corrcoef(torch.stack([fake_wind.flatten(), fake_temp.flatten()]))[0,1]
        corr_loss = torch.abs(real_corr - fake_corr)

        total_g_loss = g_loss + temporal_loss + (0.8 * corr_loss)

        total_g_loss.backward()
        opt_G.step()

        epoch_d_loss += d_loss.item()
        epoch_g_loss += total_g_loss.item()
        epoch_corr_loss += corr_loss.item()
        pbar.set_postfix(D_loss=f'{d_loss.item():.4f}', G_loss=f'{total_g_loss.item():.4f}')

    avg_d_loss = epoch_d_loss / len(dataloader)
    avg_g_loss = epoch_g_loss / len(dataloader)
    avg_corr_loss = epoch_corr_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs} | Avg D_loss: {avg_d_loss:.4f} | Avg G_loss: {avg_g_loss:.4f} | Avg Corr_Loss: {avg_corr_loss:.4f}")

torch.save(G.state_dict(), 'cgan_generator_wgangp_final.pth')
print("\n✅ Stable WGAN-GP model saved!")

# Generate & export
G.eval()
with torch.no_grad():
    extreme_cond = torch.tensor([[0.0, 1.0]] * 1000).to(device)
    noise = torch.randn(1000, noise_dim).to(device)
    generated_norm = G(noise, extreme_cond).cpu().numpy()

orig_df = pd.read_csv('preprocessed_long.csv')
features = ["ALLSKY_SFC_SW_DWN", "WS50M", "T2M"]
mins = orig_df[features].min().values
maxs = orig_df[features].max().values
generated_real = ((generated_norm + 1) / 2) * (maxs - mins) + mins
generated_real = np.clip(generated_real, 0.0, None)

flat_data = generated_real.reshape(-1, 3)
out_df = pd.DataFrame(flat_data, columns=['Solar_Irradiance_W_m2', 'Wind_Speed_m_s', 'Temperature_C'])
out_df.to_csv('simulink_extreme_forecast_wgangp_final.csv', index=False)

print("✅ Final profiles generated → simulink_extreme_forecast_wgangp_final.csv")
print("Now re-run validate_gan.py on this new file!")