#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File chạy chính để huấn luyện mô hình ResNet18 trên tập dữ liệu Imagenette2-320
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Import các module từ file huấn luyện
from train_imagenette2_by_resnet18 import (
    WNID_TO_NAME,
    ResNet18,
    train_transform,
    val_transform,
    train_model,
    plot_training_results,
    create_model_and_optimizer,
    create_data_loaders,
    get_device,
    parse_args
)

def main():
    # Phân tích tham số dòng lệnh
    args = parse_args()
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Thiết lập seed cho tái lập kết quả
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Thiết lập device (GPU/MPS nếu có)
    device = get_device()
    print(f"Sử dụng device: {device}")
    
    # Tạo dataloader
    train_dataset, train_loader, val_dataset, val_loader = create_data_loaders(
        data_root=data_root,
        batch_size=batch_size,
        train_transform=train_transform,
        val_transform=val_transform
    )
    
    # In ra thông tin về dataset
    print(f"Số lớp: {len(train_dataset.classes)}")
    print(f"Tên WNID theo thứ tự lớp: {train_dataset.classes}")
    print(f"Số lượng ảnh huấn luyện: {len(train_dataset)}")
    print(f"Số lượng ảnh kiểm thử: {len(val_dataset)}")
    
    # Ánh xạ tên lớp từ WNID
    class_names = [WNID_TO_NAME.get(wnid, wnid) for wnid in train_dataset.classes]
    print(f"Tên lớp: {class_names}")
    
    # Tạo mô hình và optimizer
    model, optimizer, criterion = create_model_and_optimizer(
        num_classes=len(train_dataset.classes),
        learning_rate=learning_rate,
        device=device
    )
    
    # Huấn luyện mô hình
    train_loss, val_loss, train_accuracy, val_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        out_dir=out_dir
    )
    
    # Vẽ đồ thị kết quả huấn luyện
    plot_training_results(train_loss, val_loss, train_accuracy, val_accuracy, epochs, out_dir)
    
    print("\nQuá trình huấn luyện đã hoàn tất!")
    print(f"Mô hình đã được lưu vào thư mục: {out_dir}")

# Thực thi chương trình nếu chạy trực tiếp
if __name__ == "__main__":
    main()
