import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
import joblib
import os
from utils import add_new_features 

# 1. Chuẩn bị dữ liệu
if not os.path.exists('model'):
    os.makedirs('model')

column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

try:
    df = pd.read_csv("data/cleveland.csv", header=None, names=column_names, na_values="?")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file data/cleveland.csv")
    exit()

# Xử lý nhãn: 0 là không bệnh, 1-4 là có bệnh
df["target"] = (df["target"] > 0).astype(int)
X = df.drop("target", axis=1)
y = df["target"]

# 2. Xây dựng Pipeline tiền xử lý
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "chol_per_age", "bps_per_age", "hr_ratio"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Điền median cho số 
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Điền mode cho category 
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ])

# 3. Pipeline hoàn chỉnh: Feature Eng -> Preprocess -> Feature Selection -> Model
# Thêm bước SelectKBest: Chọn ra K đặc trưng tốt nhất dựa trên Mutual Information 
feature_engineering_pipe = Pipeline(steps=[
    ('fe', FunctionTransformer(add_new_features, validate=False)), 
    ('preprocessor', preprocessor),
    ('selection', SelectKBest(score_func=mutual_info_classif, k=13)) # Chọn 13 đặc trưng tốt nhất
])

# 4. Định nghĩa Ensemble
estimators = [
    ('knn', KNeighborsClassifier(n_neighbors=11)),
    ('dt', DecisionTreeClassifier(max_depth=4, random_state=42)),
    ('nb', GaussianNB())
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=KNeighborsClassifier(n_neighbors=5),
    stack_method="predict_proba"
)

full_pipeline = Pipeline(steps=[
    ('processing', feature_engineering_pipe),
    ('model', stacking_model)
])

# 5. Tìm kiếm tham số tối ưu (GridSearchCV) thay vì fix cứng
# Bước này giúp tìm ra cấu hình tốt nhất cho bộ dữ liệu cụ thể này
param_grid = {
    'processing__selection__k': [10, 13, 15, 'all'], # Thử chọn số lượng đặc trưng khác nhau
    'model__knn__n_neighbors': [5, 9, 11, 15],       # Thử các giá trị K khác nhau cho KNN con
    'model__final_estimator__n_neighbors': [3, 5, 7] # Thử K cho meta-learner
}

print("Đang tìm kiếm tham số tối ưu (GridSearch)... quá trình này có thể mất 1-2 phút...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(full_pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Tham số tốt nhất: {grid_search.best_params_}")

# 6. Đánh giá và Lưu
score = best_model.score(X_test, y_test)
print(f"Độ chính xác trên tập test (Sau khi tối ưu): {score:.4f}")

joblib.dump(best_model, 'model/heart_model.pkl')
print("Đã lưu mô hình tốt nhất tại model/heart_model.pkl")