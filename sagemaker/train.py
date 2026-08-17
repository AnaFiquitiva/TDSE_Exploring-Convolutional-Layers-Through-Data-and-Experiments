"""Script de entrenamiento para Amazon SageMaker (contenedor PyTorch).

Uso esperado dentro de un SageMaker Training Job:
    python train.py --epochs 15 --batch-size 64 --lr 1e-3

Convenciones del contenedor PyTorch de SageMaker:
    - Los datos de entrenamiento/validación llegan en SM_CHANNEL_TRAIN / SM_CHANNEL_VAL
      (canales S3 declarados al llamar estimator.fit({'train': ..., 'val': ...})).
    - El modelo entrenado debe guardarse en SM_MODEL_DIR para que SageMaker lo empaquete.
    - model_fn(model_dir) es requerida por el contenedor de inferencia para reconstruir
      el modelo al desplegar el endpoint.
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

NUM_CLASSES = 10
IMAGE_SIZE = 64

# Media y desviación estándar por canal calculadas sobre EuroSAT RGB en el notebook de EDA.
EUROSAT_MEAN = [0.3444, 0.3803, 0.4078]
EUROSAT_STD = [0.2037, 0.1366, 0.1148]


class EuroSATCNN(nn.Module):
    """CNN de 3 bloques Conv3x3-ReLU-MaxPool, misma arquitectura ganadora del notebook."""

    def __init__(self, num_classes=NUM_CLASSES, base_channels=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        spatial = IMAGE_SIZE // 8
        flat_dim = base_channels * 4 * spatial * spatial
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "data/train"))
    parser.add_argument("--val-dir", type=str, default=os.environ.get("SM_CHANNEL_VAL", "data/val"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "model"))
    return parser.parse_args()


def build_dataloaders(train_dir, val_dir, batch_size):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD),
    ])
    train_ds = ImageFolder(train_dir, transform=tfm)
    val_ds = ImageFolder(val_dir, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(args.train_dir, args.val_dir, args.batch_size)

    model = EuroSATCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"epoch={epoch + 1} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


def model_fn(model_dir):
    """Requerida por el contenedor de inferencia de SageMaker para reconstruir el modelo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EuroSATCNN().to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    train(parse_args())
