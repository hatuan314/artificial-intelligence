import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchsummary import summary
import matplotlib.pyplot as plt

# Định nghĩa mô hình ResNet-18
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        # Sử dụng pre-trained weights trên ImageNet
        self.model = models.resnet18(pretrained=False, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

# Tạo mô hình
model = ResNet18(num_classes=1000)

# Hiển thị thông tin mô hình (tổng số tham số và kiến trúc)
summary(model, (3, 224, 224))  # Nhập vào hình ảnh có kích thước (3, 224, 224)

# Kiểm tra mô hình với một batch dữ liệu giả (dummy data)
dummy_input = torch.randn(1, 3, 224, 224)  # batch_size=1, channels=3, chiều cao và chiều rộng=224
output = model(dummy_input)  # Chạy qua mô hình
print(f'Output shape: {output.shape}')  # In ra shape của đầu ra

# Lưu kiến trúc mô hình ra file hình ảnh
from torchviz import make_dot

# Chạy qua mô hình để lấy đồ thị tính toán
y = model(dummy_input)
make_dot(y, params=dict(model.named_parameters())).render("resnet18_architecture", format="png")
print("Đã lưu kiến trúc mô hình vào file resnet18_architecture.png")

# Nếu bạn muốn xem ảnh của kiến trúc, dùng matplotlib:
img = plt.imread("resnet18_architecture.png")
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off')
plt.show()
