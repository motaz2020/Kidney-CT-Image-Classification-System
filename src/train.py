import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    data_dir = "processed_dataset"

    train_loader, val_loader, _, classes = get_data_loaders(data_dir)
    num_classes = len(classes)

    model = get_model(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "models/best_model.pth")


if __name__ == "__main__":
    train()