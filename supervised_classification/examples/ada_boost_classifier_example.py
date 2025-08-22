# -*- coding: utf-8 -*-
"""
Ví dụ về AdaBoost Classifier

AdaBoost (Adaptive Boosting) là một thuật toán học máy tổng hợp, kết hợp nhiều bộ phân loại yếu 
để tạo ra một bộ phân loại mạnh. Mỗi bộ phân loại mới tập trung vào các mẫu bị phân loại sai 
bởi các bộ phân loại trước đó.
"""

# Import các thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier  # Thuật toán AdaBoost
from sklearn.tree import DecisionTreeClassifier  # Bộ phân loại cây quyết định (weak learner)
from sklearn.datasets import make_classification  # Tạo dữ liệu phân loại giả
from sklearn.model_selection import train_test_split  # Chia dữ liệu thành tập huấn luyện và kiểm tra
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Đánh giá mô hình
import seaborn as sns  # Thư viện trực quan hóa

# Thiết lập seed để kết quả có thể tái tạo lại được
np.random.seed(42)

# 1. Tạo dữ liệu phân loại giả
# n_samples: số lượng mẫu
# n_features: số lượng đặc trưng
# n_informative: số lượng đặc trưng có ích cho việc phân loại
# n_redundant: số lượng đặc trưng dư thừa (không cần thiết)
# random_state: giá trị seed để tái tạo lại kết quả
X, y = make_classification(
    n_samples=1000,  # 1000 mẫu dữ liệu
    n_features=4,    # 4 đặc trưng
    n_informative=2, # Chỉ 2 đặc trưng thực sự hữu ích cho việc phân loại
    n_redundant=0,   # Không có đặc trưng dư thừa
    random_state=42, # Đặt seed để kết quả có thể tái tạo lại
    shuffle=True     # Xáo trộn dữ liệu
)

# 2. Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Kích thước tập huấn luyện: {X_train.shape[0]} mẫu")
print(f"Kích thước tập kiểm tra: {X_test.shape[0]} mẫu")

# 3. Khởi tạo và huấn luyện mô hình AdaBoost
# Sử dụng cây quyết định với độ sâu tối đa là 1 (decision stump) làm bộ phân loại yếu
base_estimator = DecisionTreeClassifier(max_depth=1)

# Khởi tạo mô hình AdaBoost
# n_estimators: số lượng bộ phân loại yếu
# learning_rate: tốc độ học (ảnh hưởng đến trọng số của mỗi bộ phân loại)
# Trong scikit-learn mới, 'base_estimator' đã được đổi thành 'estimator'
ada_boost = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# Huấn luyện mô hình trên tập huấn luyện
print("Đang huấn luyện mô hình AdaBoost...")
ada_boost.fit(X_train, y_train)
print("Hoàn thành huấn luyện!")

# 4. Đánh giá mô hình
# Dự đoán trên tập huấn luyện
y_train_pred = ada_boost.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Độ chính xác trên tập huấn luyện: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

# Dự đoán trên tập kiểm tra
y_test_pred = ada_boost.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Độ chính xác trên tập kiểm tra: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# 5. Báo cáo phân loại chi tiết
print("\nBáo cáo phân loại chi tiết:")
print(classification_report(y_test, y_test_pred))

# 6. Ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(y_test, y_test_pred)

# Trực quan hóa ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
plt.xlabel('Nhãn dự đoán')
plt.ylabel('Nhãn thực tế')

# 7. Trực quan hóa ranh giới quyết định của mô hình
# Chỉ sử dụng 2 đặc trưng đầu tiên để trực quan hóa
def plot_decision_boundary(X, y, model, title):
    # Tạo lưới điểm để vẽ ranh giới quyết định
    h = 0.02  # kích thước bước lưới
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Dự đoán nhãn cho mỗi điểm trong lưới
    Z = model.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())])
    Z = Z.reshape(xx.shape)
    
    # Vẽ ranh giới quyết định và các điểm dữ liệu
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=50)
    plt.title(title)
    plt.xlabel('Đặc trưng 1')
    plt.ylabel('Đặc trưng 2')
    plt.colorbar()

# Vẽ ranh giới quyết định cho tập kiểm tra
plot_decision_boundary(X_test, y_test, ada_boost, 'Ranh giới quyết định của AdaBoost')

# 8. Trực quan hóa tầm quan trọng của các đặc trưng
plt.figure(figsize=(10, 6))
feature_importance = ada_boost.feature_importances_
feature_names = [f'Đặc trưng {i+1}' for i in range(X.shape[1])]

# Sắp xếp các đặc trưng theo tầm quan trọng giảm dần
sorted_idx = np.argsort(feature_importance)

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Tầm quan trọng')
plt.title('Tầm quan trọng của các đặc trưng trong mô hình AdaBoost')

# 9. Ví dụ dự đoán với một mẫu mới
new_sample = np.array([[0, 0, 0, 0]])  # Mẫu mới với tất cả đặc trưng bằng 0
prediction = ada_boost.predict(new_sample)
prediction_proba = ada_boost.predict_proba(new_sample)

print(f"\nDự đoán cho mẫu mới {new_sample[0]}: Lớp {prediction[0]}")
print(f"Xác suất dự đoán: Lớp 0: {prediction_proba[0][0]:.4f}, Lớp 1: {prediction_proba[0][1]:.4f}")

# Hiển thị tất cả các biểu đồ
plt.tight_layout()
plt.show()