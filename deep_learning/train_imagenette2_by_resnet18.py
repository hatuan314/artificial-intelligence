import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import random

# Bảng ánh xạ WNID -> tên lớp người đọc
WNID_TO_NAME = {
    "n01440764": "Tench",
    "n02102040": "English Springer",
    "n02979186": "Cassette Player",
    "n03000684": "Chain Saw",
    "n03028079": "Church",
    "n03394916": "French Horn",
    "n03417042": "Garbage Truck",
    "n03425413": "Gas Pump",
    "n03445777": "Golf Ball",
    "n03888257": "Parachute",
}

# Hàm phân tích tham số dòng lệnh
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./imagenette2-320",
                   help="Thư mục chứa imagenette2-320 (bên trong có train/ và val/)")
    p.add_argument("--out-dir", type=str, default="./outputs",
                   help="Nơi lưu các kết quả và mô hình")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size cho DataLoader")
    p.add_argument("--epochs", type=int, default=10, help="Số epochs huấn luyện")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()

# Thiết lập seed cho tái lập kết quả
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Các tham số huấn luyện
args = parse_args()
batch_size = args.batch_size
learning_rate = args.lr
epochs = args.epochs
data_root = Path(args.data_root)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
set_seed(args.seed)

# Định nghĩa các phép biến đổi ảnh (resize và chuẩn hóa)
# Biến đổi cho tập huấn luyện (có tăng cường dữ liệu)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Biến đổi cho tập validation (không tăng cường)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hàm tạo dataloader
def create_data_loaders(data_root, batch_size, train_transform, val_transform):
    train_dataset = datasets.ImageFolder(root=str(data_root / "train"), transform=train_transform)
    # Sử dụng pin_memory chỉ khi sử dụng CUDA
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=use_pin_memory
    )
    
    val_dataset = datasets.ImageFolder(root=str(data_root / "val"), transform=val_transform)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=use_pin_memory
    )
    
    return train_dataset, train_loader, val_dataset, val_loader

# Định nghĩa mô hình ResNet-18
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        # Sử dụng pre-trained weights trên ImageNet
        self.model = models.resnet18(weights=None, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

# Xác định hỗ trợ GPU/MPS
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Hàm khởi tạo mô hình và optimizer
def create_model_and_optimizer(num_classes, learning_rate, device):
    model = ResNet18(num_classes=num_classes)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

# Hàm huấn luyện mô hình
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, out_dir):
    best_val_acc = 0.0
    train_loss, val_loss = [], []
    train_accuracy, val_accuracy = [], []
    
    for epoch in range(epochs):
        # Huấn luyện
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Tính toán độ chính xác và lưu kết quả
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_train / total_train
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        # Kiểm thử trên tập validation
        model.eval()
        running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_loss / len(val_loader)
        epoch_val_accuracy = 100 * correct_val / total_val
        val_loss.append(epoch_val_loss)
        val_accuracy.append(epoch_val_accuracy)

        # In ra kết quả mỗi epoch
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.2f}%')
        
        # Lưu mô hình tốt nhất
        if epoch_val_accuracy > best_val_acc:
            best_val_acc = epoch_val_accuracy
            model_path = out_dir / "best_resnet18_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'train_accuracy': epoch_accuracy,
            }, model_path)
            print(f"\u0110ã lưu mô hình tốt nhất tại epoch {epoch+1} với độ chính xác {best_val_acc:.2f}% vào {model_path}")
    
    # Lưu mô hình cuối cùng
    final_model_path = out_dir / "final_resnet18_model.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': epoch_val_accuracy,
        'train_accuracy': epoch_accuracy,
    }, final_model_path)
    print(f"\u0110ã lưu mô hình cuối cùng vào {final_model_path}")
    
    return train_loss, val_loss, train_accuracy, val_accuracy

# Hàm vẽ đồ thị kết quả huấn luyện
def plot_training_results(train_loss, val_loss, train_accuracy, val_accuracy, epochs, out_dir):
    plt.figure(figsize=(12, 6))
    
    # Tạo mảng epoch cho trục x
    epoch_range = list(range(1, epochs + 1))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    
    # Xử lý trường hợp chỉ có một epoch
    if epochs == 1:
        # Sử dụng scatter plot thay vì line plot
        plt.scatter(epoch_range, train_accuracy, label='Train Accuracy', color='blue', marker='o', s=100)
        plt.scatter(epoch_range, val_accuracy, label='Validation Accuracy', color='orange', marker='s', s=100)
        # Mở rộng trục x để đồ thị dễ nhìn hơn
        plt.xlim(0.5, 1.5)
    else:
        plt.plot(epoch_range, train_accuracy, 'o-', label='Train Accuracy')
        plt.plot(epoch_range, val_accuracy, 's-', label='Validation Accuracy')
    
    # Thiết lập giới hạn trục y cho đồ thị accuracy
    min_acc = min(min(train_accuracy), min(val_accuracy)) - 5  # Giảm 5% để có khoảng trống
    max_acc = max(max(train_accuracy), max(val_accuracy)) + 5  # Tăng 5% để có khoảng trống
    plt.ylim(max(0, min_acc), min(100, max_acc))  # Đảm bảo giới hạn hợp lý
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over epochs')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Loss plot
    plt.subplot(1, 2, 2)
    
    # Xử lý trường hợp chỉ có một epoch
    if epochs == 1:
        # Sử dụng scatter plot thay vì line plot
        plt.scatter(epoch_range, train_loss, label='Train Loss', color='blue', marker='o', s=100)
        plt.scatter(epoch_range, val_loss, label='Validation Loss', color='orange', marker='s', s=100)
        # Mở rộng trục x để đồ thị dễ nhìn hơn
        plt.xlim(0.5, 1.5)
    else:
        plt.plot(epoch_range, train_loss, 'o-', label='Train Loss')
        plt.plot(epoch_range, val_loss, 's-', label='Validation Loss')
    
    # Thiết lập giới hạn trục y cho đồ thị loss
    min_loss = min(min(train_loss), min(val_loss)) * 0.9  # Giảm 10% để có khoảng trống
    max_loss = max(max(train_loss), max(val_loss)) * 1.1  # Tăng 10% để có khoảng trống
    plt.ylim(min_loss, max_loss)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over epochs')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    # Lưu đồ thị
    plot_path = out_dir / "training_results.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\u0110ã lưu đồ thị kết quả huấn luyện vào {plot_path}")
    
    # Hiển thị đồ thị
    plt.show()

# Nếu chạy trực tiếp file này
if __name__ == "__main__":
    print("\nFile này chứa các hàm và lớp cho việc huấn luyện ResNet18 trên Imagenette.")
    print("\u0110ể chạy huấn luyện, hãy sử dụng file main.py:\n")
    print("python main.py --epochs 10 --batch-size 32 --lr 0.001\n")
