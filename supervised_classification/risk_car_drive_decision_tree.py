from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Dữ liệu training
train_data = {
    'time': ['1-2', '2-7', '>7', '1-2', '>7', '1-2', '2-7', '2-7'],
    'gender': ['m', 'm', 'f', 'f', 'm', 'm', 'f', 'm'],
    'area': ['urban', 'rural', 'rural', 'rural', 'rural', 'rural', 'urban', 'urban'],
    'risk': ['low', 'high', 'high', 'high', 'high', 'high', 'low', 'low']
}

df = pd.DataFrame(train_data)

# Encode các cột dạng text
encoders = {}
for col in df.columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# Tách X, y
X = df.drop('risk', axis=1)
y = df['risk']

# Huấn luyện mô hình
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

# Hiển thị cây quyết định
print(export_text(model, feature_names=X.columns.tolist()))

# Dữ liệu test (A, B, C)
test_data = pd.DataFrame({
    'time': ['1-2', '2-7', '1-2'],
    'gender': ['f', 'm', 'f'],
    'area': ['rural', 'urban', 'urban']
})

# Encode giống training
for col in test_data.columns:
    test_data[col] = encoders[col].transform(test_data[col])

# Dự đoán
predictions = model.predict(test_data)
predicted_labels = encoders['risk'].inverse_transform(predictions)

# In kết quả
for idx, label in zip(['A', 'B', 'C'], predicted_labels):
    print(f"Dự đoán của {idx}: {label}")
