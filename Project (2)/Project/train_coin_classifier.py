
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


DATA_FOLDER     = "dataset"
TRAIN_FOLDER    = os.path.join(DATA_FOLDER, "train")
VALID_FOLDER    = os.path.join(DATA_FOLDER, "val")
IMAGE_SIZE      = 64
BATCH_SIZE      = 32
NUM_EPOCHS      = 20
LEARNING_RATE   = 0.001
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  Define transforms (smoothing + normalization)
mean_vals = [0.5, 0.5, 0.5]
std_vals  = [0.5, 0.5, 0.5]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.GaussianBlur(kernel_size=5),       # smoothing
    transforms.RandomHorizontalFlip(),            # simple augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean_vals, std_vals),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor(),
    transforms.Normalize(mean_vals, std_vals),
])

#Load datasets
training_dataset   = datasets.ImageFolder(TRAIN_FOLDER, transform=train_transform)
validation_dataset = datasets.ImageFolder(VALID_FOLDER, transform=val_transform)

# Save the class→index map so inference can use it
with open("class_to_idx.json", "w") as f:
    json.dump(training_dataset.class_to_idx, f)

#  Create data loaders
train_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(validation_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# Build a simple CNN model
class CoinClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CoinClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # after 3 pools: IMAGE_SIZE→IMAGE_SIZE/8
        flat_size = 128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#Initialize
num_classes = len(training_dataset.classes)
model       = CoinClassifier(num_classes).to(DEVICE)
criterion   = nn.CrossEntropyLoss()
optimizer   = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(1, NUM_EPOCHS + 1):
    # -- train phase
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss     += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / len(training_dataset)
    epoch_acc  = running_corrects / len(training_dataset)

    #validation phase
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss      += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects  += torch.sum(preds == labels.data).item()

    val_loss = val_loss / len(validation_dataset)
    val_acc  = val_corrects / len(validation_dataset)

    print(f"Epoch {epoch}/{NUM_EPOCHS} — "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# 7) Save the model
torch.save(model.state_dict(), "coin_classifier.pth")
print("✔ Training complete, model saved to coin_classifier.pth")
