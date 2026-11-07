
# 1. Импорты и девайс

import os, zipfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from google.colab import drive
drive.mount('/content/drive')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print(f'Устройство: {device}')



# 2. Распаковка датасетов

extract_base = '/content/dataset/'
os.makedirs(extract_base, exist_ok=True)

def unzip(name, target_name):
    target = os.path.join(extract_base, target_name)

    if os.path.exists(target) and os.listdir(target):
        print(f'{target_name}: уже распаковано ({len(os.listdir(target))} файлов)')
        return target

    os.makedirs(target, exist_ok=True)
    zpath = f'/content/drive/MyDrive/{name}'
    print(f'Распаковка {name} → {target_name}/')

    with zipfile.ZipFile(zpath, 'r') as zf:
        for n in tqdm(zf.namelist(), desc=target_name):
            zf.extract(n, target)

    print(f'  → {target_name}: {len(os.listdir(target))} файлов')
    return target

train_input_dir  = unzip('X_train_holo.zip', 'X_train_holo')
test_input_dir   = unzip('X_test_holo.zip',  'X_test_holo')
train_target_dir = unzip('X_train.zip',      'X_train')
test_target_dir  = unzip('X_test.zip',       'X_test')

print('\nВсе датасеты готовы.')



# 3. Датасет 512x512

class PairDataset512(Dataset):
    def __init__(self, input_dir, target_dir):
        def find_images(path):
            files = []
            for root, _, names in os.walk(path):
                files.extend(os.path.join(root, f) for f in names if f.endswith('.png'))
            return files

        in_map  = {os.path.splitext(os.path.basename(p))[0]: p for p in find_images(input_dir)}
        tgt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in find_images(target_dir)}

        common = sorted(set(in_map) & set(tgt_map))
        assert common, 'Нет совпадающих имён между input и target!'

        self.pairs = [(in_map[n], tgt_map[n]) for n in common]
        print(f'Найдено пар: {len(self.pairs)}')

    def __len__(self):
        return len(self.pairs)

  #  обрезаем до 256×256
    def __getitem__(self, idx):
        in_path, tgt_path = self.pairs[idx]
        x = cv2.imread(in_path,  cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        y = cv2.imread(tgt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        y = y[:256, :256]

        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y).unsqueeze(0)


full_train = PairDataset512(train_input_dir, train_target_dir)
full_test  = PairDataset512(test_input_dir,  test_target_dir)

TRAIN_SIZE = min(500, len(full_train))
TEST_SIZE  = min(500,  len(full_test))

train_loader = DataLoader(Subset(full_train, range(TRAIN_SIZE)),
                          batch_size=5, shuffle=True,  num_workers=2)
test_loader  = DataLoader(Subset(full_test,  range(TEST_SIZE)),
                          batch_size=5, shuffle=False, num_workers=2)



# 4. Модель HoloUNet (выход 256x256)
import torch.nn.functional as F

class HoloUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()

        # Space to depth: 1 канал -> 4, размер / 2
        self.s2d = nn.PixelUnshuffle(2)

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=4, padding='same'),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, padding='same'),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=4, padding='same'),
            nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, padding='same'),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=4, padding='same'),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=4, padding='same'),
            nn.ReLU(inplace=True),
        )

        # Финальная свёртка — выход 256x256
        self.final = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, x):
        # x: (B, 1, 512, 512)

        x = self.s2d(x)                      # (B, 4, 256, 256)
        e1 = self.enc1(x)                    # (B, 16, 256, 256)
        e2 = self.enc2(e1)                   # (B, 32, 256, 256)   ← SKIP
        p  = self.pool(e2)                   # (B, 32, 128, 128)
        e3 = self.enc3(p)                    # (B, 32, 128, 128)
        e4 = self.enc4(e3)                   # (B, 64, 128, 128)

        d1 = self.up(e4)                     # (B, 64, 256, 256)
        d1 = torch.cat([d1, e2], dim=1)      # (B, 96, 256, 256)
        d1 = self.dec1(d1)                   # (B, 32, 256, 256)
        d2 = self.dec2(d1)                   # (B, 32, 256, 256)

        out = self.final(d2)                 # (B, 1, 256, 256)
        return torch.sigmoid(out)


# Пересоздаём модель
import gc
gc.collect()
torch.cuda.empty_cache()

model = HoloUNet().to(device)
print(f'Параметров: {sum(p.numel() for p in model.parameters()):,}')

# 5. Обучение с чекпоинтом

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

checkpoint_path = '/content/drive/MyDrive/checkpoint_holo.pth'
TOTAL_EPOCHS = 6
start_epoch, train_losses, test_losses = 0, [], []

if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch  = ckpt['epoch']
    train_losses = ckpt['train_losses']
    test_losses  = ckpt['test_losses']
    print(f'Загружен чекпоинт: эпоха {start_epoch}')


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += criterion(model(x), y).item()
    return total / len(loader)


for epoch in range(start_epoch, TOTAL_EPOCHS):
    model.train()
    running = 0.0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{TOTAL_EPOCHS}')
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.6f}')

    train_losses.append(running / len(train_loader))
    test_losses.append(evaluate(model, test_loader))
    print(f'Epoch {epoch+1}: train={train_losses[-1]:.6f}  test={test_losses[-1]:.6f}')

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses':  test_losses,
    }, checkpoint_path)



# 6. График и сохранение

plt.figure(figsize=(10, 5))
plt.plot(train_losses, 'b-', label='Train')
plt.plot(test_losses,  'r-', label='Test')
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.grid(True)
plt.show()

torch.save(model.state_dict(), '/content/drive/MyDrive/model_holo.pth')
print('Готово.')