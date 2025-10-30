import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. Tải dữ liệu ---
try:
    pos_data = scipy.io.loadmat('possamples.mat')['possamples']
    neg_data = scipy.io.loadmat('negsamples.mat')['negsamples']

    # --- 2. Định dạng lại dữ liệu (Flattening) ---
    # Chuyển đổi từ (width, height, num_samples) sang (num_samples, width, height)
    pos_samples = np.transpose(pos_data, (2, 0, 1))
    neg_samples = np.transpose(neg_data, (2, 0, 1))

    # Chuyển đổi mỗi ảnh 2D thành một vector 1D (24*24 = 576)
    num_pos_samples = pos_samples.shape[0]
    num_neg_samples = neg_samples.shape[0]
    feature_length = pos_samples.shape[1] * pos_samples.shape[2]

    pos_vectors = pos_samples.reshape(num_pos_samples, feature_length)
    neg_vectors = neg_samples.reshape(num_neg_samples, feature_length)

    # --- 3. Tạo nhãn và gộp dữ liệu ---
    # Nhãn 1 cho 'positive' (khuôn mặt), -1 cho 'negative' (không phải khuôn mặt)
    pos_labels = np.ones(num_pos_samples)
    neg_labels = -np.ones(num_neg_samples)

    # Gộp thành một bộ dữ liệu duy nhất
    X = np.vstack((pos_vectors, neg_vectors))
    y = np.hstack((pos_labels, neg_labels))

    # --- 4. Chuẩn hóa dữ liệu ---
    # Áp dụng Mean-variance normalization (hay còn gọi là StandardScaler)
    # Nó sẽ biến đổi dữ liệu sao cho có giá trị trung bình là 0 và độ lệch chuẩn là 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(np.float64))

    # --- 5. Chia dữ liệu thành tập Huấn luyện (Train) và Xác thực (Validation) ---
    # Chia 80% cho training và 20% cho validation
    # random_state để đảm bảo kết quả chia là như nhau mỗi lần chạy
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Định dạng lại và chuẩn hóa dữ liệu thành công!")
    print("-" * 30)
    print(f"Kích thước tập dữ liệu X (đã chuẩn hóa): {X_scaled.shape}")
    print(f"Kích thước tập nhãn y: {y.shape}")
    print("-" * 30)
    print("Sau khi chia (80/20):")
    print(f"Kích thước X_train: {X_train.shape}")
    print(f"Kích thước y_train: {y_train.shape}")
    print(f"Kích thước X_val:   {X_val.shape}")
    print(f"Kích thước y_val:   {y_val.shape}")

except FileNotFoundError:
    print("Lỗi: Không tìm thấy tệp possamples.mat hoặc negsamples.mat.")
except Exception as e:
    print(f"Có lỗi xảy ra: {e}")