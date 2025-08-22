from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Dữ liệu thô
data = [
    ['<=30', 'High', 'No', 'Fair', 'No'],
    ['<=30', 'High', 'No', 'Excellent', 'No'],
    ['31...40', 'High', 'No', 'Fair', 'Yes'],
    ['>40', 'Medium', 'No', 'Fair', 'Yes'],
    ['>40', 'Low', 'Yes', 'Fair', 'Yes'],
    ['>40', 'Low', 'Yes', 'Excellent', 'No'],
    ['31...40', 'Low', 'Yes', 'Excellent', 'Yes'],
    ['<=30', 'Medium', 'No', 'Fair', 'No'],
    ['<=30', 'Low', 'Yes', 'Fair', 'Yes'],
    ['>40', 'Medium', 'Yes', 'Fair', 'Yes'],
    ['<=30', 'Medium', 'Yes', 'Excellent', 'Yes'],
    ['31...40', 'Medium', 'No', 'Excellent', 'Yes'],
    ['31...40', 'High', 'Yes', 'Fair', 'Yes'],
    ['>40', 'Medium', 'No', 'Excellent', 'No']
]

# Chia thuộc tính X và nhãn y
X_raw = [row[:-1] for row in data]
y_raw = [row[-1] for row in data]

# Mã hóa Label
le_age = LabelEncoder()
le_income = LabelEncoder()
le_student = LabelEncoder()
le_credit = LabelEncoder()
le_label = LabelEncoder()

X_encoded = np.array([
    le_age.fit_transform([x[0] for x in X_raw]),
    le_income.fit_transform([x[1] for x in X_raw]),
    le_student.fit_transform([x[2] for x in X_raw]),
    le_credit.fit_transform([x[3] for x in X_raw])
]).T

y_encoded = le_label.fit_transform(y_raw)

# Huấn luyện mô hình Naive Bayes
model = CategoricalNB()
model.fit(X_encoded, y_encoded)

# Mẫu cần dự đoán: Age <= 30, Income = Medium, Student = Yes, Credit = Fair
test_sample = [
    le_age.transform(['<=30'])[0],
    le_income.transform(['Medium'])[0],
    le_student.transform(['Yes'])[0],
    le_credit.transform(['Fair'])[0]
]

predicted = model.predict([test_sample])
predicted_label = le_label.inverse_transform(predicted)

print("Dự đoán:", predicted_label[0])
