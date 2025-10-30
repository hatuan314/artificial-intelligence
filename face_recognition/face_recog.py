# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# ===================================================================
# PHẦN 1 & 2: HUẤN LUYỆN MÔ HÌNH
# ===================================================================
def train_face_detector():
    """
    Tải dữ liệu, chuẩn hóa và huấn luyện mô hình SVM.
    Hàm này giả định các tệp .mat nằm trong cùng thư mục.
    """
    print("Bắt đầu huấn luyện mô hình...")
    
    # Tải dữ liệu từ các tệp .mat
    pos_data = scipy.io.loadmat('possamples.mat')['possamples']
    neg_data = scipy.io.loadmat('negsamples.mat')['negsamples']
    
    # Định dạng lại dữ liệu: chuyển mỗi ảnh 24x24 thành vector 576 chiều
    pos_samples = np.transpose(pos_data, (2, 0, 1)).reshape(pos_data.shape[2], -1)
    neg_samples = np.transpose(neg_data, (2, 0, 1)).reshape(neg_data.shape[2], -1)
    
    # Gộp dữ liệu và tạo nhãn (1 cho khuôn mặt, -1 cho không phải khuôn mặt)
    X = np.vstack((pos_samples, neg_samples))
    y = np.hstack((np.ones(pos_samples.shape[0]), -np.ones(neg_samples.shape[0])))
    
    # Chuẩn hóa dữ liệu (quan trọng cho SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(np.float64))
    
    # Chia dữ liệu (không cần thiết nếu huấn luyện trên toàn bộ, nhưng giữ để nhất quán)
    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Huấn luyện mô hình SVM tuyến tính
    # dual=False được khuyên dùng khi số lượng mẫu > số lượng đặc trưng
    svm_model = LinearSVC(C=1.5, dual=False, random_state=42, max_iter=10000)
    svm_model.fit(X_train, y_train)
    
    print("Huấn luyện hoàn tất!")
    return svm_model, scaler

# ===================================================================
# PHẦN 3: CÁC HÀM XỬ LÝ ẢNH VÀ NHẬN DIỆN
# ===================================================================

def scanning_window_detection(image, model, scaler, step_size=8, window_size=(24, 24), conf_threshold=0.5):
    """
    Quét ảnh bằng cửa sổ trượt, trích xuất các vùng và dự đoán bằng SVM.
    """
    detections = []
    img_w, img_h = image.size
    win_w, win_h = window_size
    
    # Chuyển ảnh sang ảnh xám để xử lý
    gray_image = image.convert('L')
    
    # Lặp qua toàn bộ ảnh với một bước nhảy (step_size)
    for y in range(0, img_h - win_h + 1, step_size):
        for x in range(0, img_w - win_w + 1, step_size):
            # Trích xuất vùng ảnh (patch)
            patch = gray_image.crop((x, y, x + win_w, y + win_h))
            patch_vector = np.array(patch).flatten().reshape(1, -1)
            
            # Chuẩn hóa vector bằng scaler đã được huấn luyện
            patch_scaled = scaler.transform(patch_vector)
            
            # Sử dụng decision_function để lấy điểm tin cậy
            score = model.decision_function(patch_scaled)[0]
            
            # Nếu điểm tin cậy vượt ngưỡng, lưu lại tọa độ và điểm số
            if score > conf_threshold:
                detections.append((x, y, x + win_w, y + win_h, score))
                
    return detections

def non_max_suppression(boxes, overlap_thresh=0.3):
    """
    Thực hiện Non-maxima Suppression để lọc các hộp giới hạn trùng lặp.
    Chỉ giữ lại những hộp có điểm tin cậy cao nhất và không trùng lặp nhiều.
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1, y1, x2, y2, score = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(score)
    
    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Tìm tọa độ của hộp giao (intersection)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # Tính tỷ lệ chồng chéo (Intersection over Union - IoU)
        overlap = (w * h) / area[idxs[:last]]
        
        # Xóa các hộp có tỷ lệ chồng chéo lớn hơn ngưỡng
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
        
    return boxes[pick].astype("int")

# ===================================================================
# CHƯƠNG TRÌNH CHÍNH
# ===================================================================
if __name__ == "__main__":
    # 1. Huấn luyện mô hình (chỉ chạy một lần)
    model, scaler = train_face_detector()
    
    # 2. Xử lý tất cả các ảnh trong thư mục 'img'
    image_folder = 'img'
    
    if not os.path.isdir(image_folder):
        print(f"\nLỖI: Không tìm thấy thư mục '{image_folder}'.")
        print("Vui lòng tạo thư mục 'img' và đặt các ảnh của bạn vào đó.")
    else:
        # Lặp qua tất cả các tệp trong thư mục
        for image_name in os.listdir(image_folder):
            # Chỉ xử lý các tệp có đuôi là ảnh
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_image_path = os.path.join(image_folder, image_name)
                
                image = Image.open(test_image_path)
                
                # 3. Quét ảnh để tìm các phát hiện thô
                print(f"\nĐang quét ảnh '{test_image_path}'...")
                raw_detections = scanning_window_detection(image, model, scaler, conf_threshold=0.8)
                print(f"Tìm thấy {len(raw_detections)} phát hiện thô.")
                
                # 4. Áp dụng NMS để lọc kết quả
                print("Đang áp dụng Non-maxima Suppression...")
                final_boxes = non_max_suppression(raw_detections, overlap_thresh=0.2)
                print(f"Còn lại {len(final_boxes)} khuôn mặt sau khi lọc.")
                
                # 5. Vẽ kết quả lên ảnh
                draw = ImageDraw.Draw(image)
                for (x1, y1, x2, y2, score) in final_boxes:
                    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
                
                # 6. Hiển thị ảnh kết quả
                plt.figure(figsize=(8, 8))
                plt.imshow(image)
                plt.title(f"Kết quả nhận diện trên '{image_name}'")
                plt.axis('off')
                plt.show() # Hiển thị ảnh, chương trình sẽ tạm dừng ở đây cho đến khi bạn đóng cửa sổ ảnh
