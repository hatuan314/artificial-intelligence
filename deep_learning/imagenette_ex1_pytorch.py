#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bài tập 1 - Imagenette2-320 (PyTorch)
-------------------------------------
Mục tiêu (train set):
1) Vẽ biểu đồ cột số lượng ảnh của mỗi lớp và nhận xét.
2) Mỗi lớp hiển thị 3 ảnh mẫu + nhãn (để trực quan hoá).
3) Tăng cường dữ liệu (Horizontal Flip, Random Rotation [-15, 15], Brightness/Contrast) và hiển thị kết quả.

Hướng dẫn nhanh:
- Yêu cầu: Python 3.9+, pip install torch torchvision matplotlib pillow
- Tải và giải nén bộ dữ liệu "imagenette2-320" vào thư mục: ./data/imagenette2-320/
  Cấu trúc mong đợi:
    data/imagenette2-320/
      train/
        n01440764/
        n02102040/
        ...
      val/
        n01440764/
        n02102040/
        ...
- Chạy: python imagenette_ex1_pytorch.py --data-root ./data/imagenette2-320 --out-dir ./outputs

File sẽ sinh ra:
- class_counts.png        : Biểu đồ số lượng ảnh theo lớp.
- samples_per_class.png   : 3 ảnh/mỗi lớp (từ train).
- augmentations_grid.png  : Minh hoạ tăng cường dữ liệu (gốc + 3 biến đổi).
"""

import argparse
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils


# Bảng ánh xạ WNID -> tên lớp người đọc (tham khảo đề bài)
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./imagenette2-320",
                   help="Thư mục chứa imagenette2-320 (bên trong có train/ và val/)")
    p.add_argument("--out-dir", type=str, default="./outputs",
                   help="Nơi lưu các hình kết quả")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size cho DataLoader")
    return p.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # đảm bảo một chút tái lập, nhưng không cần quá cứng nhắc cho phần trực quan hoá


def check_data_structure(data_root: Path) -> bool:
    """Kiểm tra xem đã có thư mục train/ và ít nhất một lớp con chưa."""
    train_dir = data_root / "train"
    if not train_dir.exists():
        print(f"[!] Không tìm thấy thư mục train/ tại: {train_dir.resolve()}")
        return False
    # ít nhất một lớp (WNID) có ảnh
    subdirs = [p for p in train_dir.iterdir() if p.is_dir()]
    if not subdirs:
        print(f"[!] Không tìm thấy các thư mục lớp (WNID) bên trong: {train_dir.resolve()}")
        return False
    return True


def build_datasets_and_loaders(data_root: Path, batch_size: int = 32):
    """Tạo Dataset & DataLoader cho tập train (resize về 224x224 để nhất quán với các bài sau)."""
    # Biến đổi CƠ BẢN (không tăng cường) để đếm/hiển thị ảnh gốc
    base_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Dùng ImageFolder: mỗi thư mục con là 1 lớp
    train_ds = datasets.ImageFolder(root=str(data_root / "train"), transform=base_tfms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    return train_ds, train_loader


def count_images_per_class(train_ds: datasets.ImageFolder):
    """Đếm số ảnh theo nhãn (chỉ số), sau đó quy về WNID và tên lớp."""
    # train_ds.class_to_idx: {'n01440764': 0, ...}
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    # train_ds.samples: list[(path, class_idx)]
    counts = Counter([cls_idx for _, cls_idx in train_ds.samples])

    # Map thành (WNID -> count) và (Tên -> count) để vẽ đẹp mắt
    wnids = [idx_to_class[i] for i in range(len(idx_to_class))]
    counts_by_wnid = {wnid: counts[i] for wnid, i in train_ds.class_to_idx.items()}
    names_in_order = [WNID_TO_NAME.get(wnid, wnid) for wnid in wnids]
    counts_in_order = [counts_by_wnid[wnid] for wnid in wnids]
    return wnids, names_in_order, counts_in_order


def plot_class_counts(names_in_order, counts_in_order, out_path: Path):
    plt.figure(figsize=(12, 5))
    plt.bar(names_in_order, counts_in_order)
    plt.title("Số lượng ảnh theo lớp (train)")
    plt.ylabel("Số ảnh")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[✓] Đã lưu biểu đồ: {out_path}")


def select_three_images_per_class(train_ds: datasets.ImageFolder):
    """Chọn ngẫu nhiên 3 ảnh cho mỗi lớp, trả về danh sách các tensor ảnh và nhãn chữ."""
    # Gom các index theo class_idx
    per_class_indices = defaultdict(list)
    for i, (_, cls_idx) in enumerate(train_ds.samples):
        per_class_indices[cls_idx].append(i)

    images, labels = [], []
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    for cls_idx, indices in per_class_indices.items():
        # đảm bảo có ít nhất 3, nếu ít thì lấy min(len, 3)
        k = min(3, len(indices))
        chosen = random.sample(indices, k=k)
        for ds_index in chosen:
            img, _ = train_ds[ds_index]  # đã transform -> Tensor [3,224,224]
            wnid = idx_to_class[cls_idx]
            label = WNID_TO_NAME.get(wnid, wnid)
            images.append(img)
            labels.append(label)
    return images, labels


def make_grid_with_captions(images, labels, ncols=3, out_path: Path = None, title="3 ảnh mỗi lớp (train)"):
    """Vẽ lưới ảnh với chú thích nhãn. Mặc định 3 cột -> mỗi lớp một hàng."""
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=(3 * ncols + 2, 3 * nrows + 2))
    for i, (img, lbl) in enumerate(zip(images, labels), start=1):
        plt.subplot(nrows, ncols, i)
        # img là Tensor [C,H,W] -> [H,W,C]
        npimg = img.permute(1, 2, 0).numpy()
        plt.imshow(npimg)
        plt.axis("off")
        plt.title(lbl, fontsize=9)
    plt.suptitle(title)
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"[✓] Đã lưu ảnh lưới: {out_path}")
    plt.close()


def build_aug_transforms():
    """Tạo 3 phép tăng cường theo yêu cầu + Resize/ToTensor để demo."""
    # 1) Lật ngang
    aug_flip = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),  # p=1 để chắc chắn thấy khác biệt
        transforms.ToTensor(),
    ])
    # 2) Xoay [-15, 15]
    aug_rotate = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ])
    # 3) Brightness/Contrast
    aug_bc = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.ToTensor(),
    ])
    # Pipeline tổng hợp (có thể dùng khi train thật)
    aug_combo = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
    ])
    return aug_flip, aug_rotate, aug_bc, aug_combo


def demo_augmentations(train_root: Path, out_path: Path, max_classes: int = 10):
    """Minh hoạ: chọn 1 ảnh từ mỗi lớp -> tạo 1 hàng gồm: [gốc, flip, rotate, jitter]."""
    flip, rotate, jitter, _ = build_aug_transforms()
    base = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # Lấy 1 ảnh gốc/ lớp
    rows = []
    captions = []  # tiêu đề cho mỗi cột
    wnid_dirs = sorted([d for d in (train_root).iterdir() if d.is_dir()])
    wnid_dirs = wnid_dirs[:max_classes]

    for wnid_dir in wnid_dirs:
        # tìm 1 file ảnh bất kỳ trong lớp
        img_files = [p for p in wnid_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if not img_files:
            continue
        img_path = random.choice(img_files)
        pil = Image.open(img_path).convert("RGB")
        goc = base(pil)
        f1 = flip(pil)
        r1 = rotate(pil)
        j1 = jitter(pil)
        rows.extend([goc, f1, r1, j1])
        # tạo nhãn hàng
        wnid = wnid_dir.name
        name = WNID_TO_NAME.get(wnid, wnid)
        captions.append(name)

    # Vẽ lưới: mỗi hàng 4 ảnh (gốc + 3 augment)
    ncols = 4
    nrows = len(captions)
    plt.figure(figsize=(4 * ncols + 2, 3 * nrows + 2))
    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c
            plt.subplot(nrows, ncols, idx + 1)
            npimg = rows[idx].permute(1, 2, 0).numpy()
            plt.imshow(npimg)
            plt.axis("off")
            if r == 0:
                # tiêu đề cột
                if c == 0:
                    plt.title("Gốc", fontsize=10)
                elif c == 1:
                    plt.title("Flip ngang", fontsize=10)
                elif c == 2:
                    plt.title("Rotate ±15°", fontsize=10)
                else:
                    plt.title("Brightness/Contrast", fontsize=10)
            if c == 0:
                plt.ylabel(captions[r], rotation=90, fontsize=9)
    plt.suptitle("Minh hoạ tăng cường dữ liệu (mỗi hàng = 1 lớp)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[✓] Đã lưu minh hoạ augmentations: {out_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = check_data_structure(data_root)
    if not ok:
        print("\nHướng dẫn tải dữ liệu (ví dụ):")
        print("- Tải gói 'imagenette2-320' từ fastai và giải nén vào ./data/imagenette2-320/")
        print("- Đảm bảo có thư mục con: train/ và val/ bên trong.")
        print("Sau đó chạy lại lệnh:")
        print(f"  python {Path(__file__).name} --data-root ./data/imagenette2-320 --out-dir ./outputs\n")
        return

    # 1) Dataset & DataLoader
    train_ds, train_loader = build_datasets_and_loaders(data_root, batch_size=args.batch_size)
    print(f"[i] Số lớp (train): {len(train_ds.classes)}")
    print(f"[i] Tên WNID theo thứ tự lớp: {train_ds.classes}")

    # 2) Vẽ biểu đồ cột số lượng ảnh theo lớp
    wnids, names_in_order, counts_in_order = count_images_per_class(train_ds)
    plot_class_counts(names_in_order, counts_in_order, out_dir / "class_counts.png")

    # 3) Hiển thị 3 ảnh/ lớp (grid)
    imgs, lbls = select_three_images_per_class(train_ds)
    make_grid_with_captions(imgs, lbls, ncols=3, out_path=out_dir / "samples_per_class.png",
                            title="3 ảnh mẫu/ lớp (train)")

    # 4) Tăng cường dữ liệu: minh hoạ (gốc + 3 biến đổi) cho mỗi lớp
    demo_augmentations(data_root / "train", out_dir / "augmentations_grid.png")

    print("\n[Hoàn tất] Mời mở thư mục outputs/ để xem các hình kết quả:")
    print(f"- {out_dir / 'class_counts.png'}")
    print(f"- {out_dir / 'samples_per_class.png'}")
    print(f"- {out_dir / 'augmentations_grid.png'}\n")

    # Gợi ý nhận xét (in ra console) — sinh viên nên tự quan sát biểu đồ để kết luận:
    print("Gợi ý nhận xét:")
    print("- Nhìn vào class_counts.png: lớp nào nhiều/ít ảnh nhất? Bộ dữ liệu có cân bằng không?")
    print("- Quan sát samples_per_class.png: ảnh có đa dạng góc chụp, nền, ánh sáng không? Có nhiễu/blur không?")
    print("- Quan sát augmentations_grid.png: các biến đổi có giúp đa dạng hoá dữ liệu hợp lý không? Quá mạnh/yếu?")
    print("- Gợi ý khi train thật: dùng pipeline aug_combo (trong build_aug_transforms) cho train, còn val chỉ Resize+ToTensor.")
    print("- Nếu lớp quá lệch, có thể cân nhắc WeightedRandomSampler hoặc class weights (khi huấn luyện).")


if __name__ == "__main__":
    main()
