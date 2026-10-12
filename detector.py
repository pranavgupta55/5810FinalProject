"""Extremely basic single-object detection CNN (PyTorch, ReLU, Adam).

Each image -> one class label + one bounding box (cx, cy, w, h), all box
values normalized to [0, 1] relative to image size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# ---- Config (placeholder dimensions; adjust once dataset is known) ----
IMG_SIZE = 128
IN_CHANNELS = 3
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
BOX_LOSS_WEIGHT = 5.0


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class TinyDetector(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES):
        super().__init__()
        # 128 -> 64 -> 32 -> 16 -> 8
        self.features = nn.Sequential(
            conv_block(in_channels, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)  # -> 128 x 4 x 4, size-independent
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.cls_head = nn.Linear(256, num_classes)
        self.box_head = nn.Linear(256, 4)

    def forward(self, x):
        x = self.fc(self.pool(self.features(x)))
        return self.cls_head(x), torch.sigmoid(self.box_head(x))


def detection_loss(cls_logits, box_pred, labels, boxes):
    cls_loss = F.cross_entropy(cls_logits, labels)
    box_loss = F.smooth_l1_loss(box_pred, boxes)
    return cls_loss + BOX_LOSS_WEIGHT * box_loss, cls_loss, box_loss


def box_iou(a, b):
    """IoU between matching rows of two (N, 4) cx,cy,w,h tensors."""
    a1, a2 = a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2
    b1, b2 = b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(min=0).prod(1)
    union = a[:, 2:].prod(1) + b[:, 2:].prod(1) - inter
    return inter / union.clamp(min=1e-6)


class SyntheticDataset(Dataset):
    """Placeholder: random noise images with a bright rectangle. Replace with real loader."""

    def __init__(self, n=2000):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        img = torch.rand(IN_CHANNELS, IMG_SIZE, IMG_SIZE, generator=g) * 0.3
        w, h = (torch.rand(2, generator=g) * 0.4 + 0.1).tolist()
        cx = torch.rand(1, generator=g).item() * (1 - w) + w / 2
        cy = torch.rand(1, generator=g).item() * (1 - h) + h / 2
        label = idx % NUM_CLASSES
        x0, x1 = int((cx - w / 2) * IMG_SIZE), int((cx + w / 2) * IMG_SIZE)
        y0, y1 = int((cy - h / 2) * IMG_SIZE), int((cy + h / 2) * IMG_SIZE)
        img[label % IN_CHANNELS, y0:y1, x0:x1] = 1.0
        return img, label, torch.tensor([cx, cy, w, h])


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, loss_sum, correct, iou_sum = 0, 0.0, 0, 0.0
    with torch.set_grad_enabled(training):
        for imgs, labels, boxes in loader:
            imgs, labels, boxes = imgs.to(device), labels.to(device), boxes.to(device)
            cls_logits, box_pred = model(imgs)
            loss, _, _ = detection_loss(cls_logits, box_pred, labels, boxes)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            n = imgs.size(0)
            total += n
            loss_sum += loss.item() * n
            correct += (cls_logits.argmax(1) == labels).sum().item()
            iou_sum += box_iou(box_pred, boxes).sum().item()
    return loss_sum / total, correct / total, iou_sum / total


def main():
    device = get_device()
    print(f"Device: {device}")

    dataset = SyntheticDataset(2000)
    n_val = len(dataset) // 10
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = TinyDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    best_iou = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train loss {tr[0]:.4f} acc {tr[1]:.3f} iou {tr[2]:.3f}"
              f" | val loss {va[0]:.4f} acc {va[1]:.3f} iou {va[2]:.3f}")
        if va[2] > best_iou:
            best_iou = va[2]
            torch.save(model.state_dict(), "detector.pt")
    print(f"Best val IoU: {best_iou:.3f} (saved to detector.pt)")


if __name__ == "__main__":
    main()
