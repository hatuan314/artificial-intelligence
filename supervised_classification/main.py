from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dữ liệu mẫu
emails = ["Free money now!", "Hello friend", "Win a free iPhone", "Let's meet tomorrow"]
labels = [1, 0, 1, 0]  # 1 = spam, 0 = not spam

# Bước 1: Vector hóa văn bản thành số
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Bước 2: Tách tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5)

# Bước 3: Huấn luyện mô hình Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Bước 4: Dự đoán
y_pred = model.predict(X_test)

# Bước 5: Đánh giá độ chính xác
print("Độ chính xác:", accuracy_score(y_test, y_pred))