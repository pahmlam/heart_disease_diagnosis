import pandas as pd

def add_new_features(X_df):
    X_df = X_df.copy()
    # Thêm đặc trưng mới dựa trên kiến thức y học
    # Lưu ý: Cần kiểm tra cột tồn tại để tránh lỗi
    if "chol" in X_df.columns and "age" in X_df.columns:
        X_df["chol_per_age"] = X_df["chol"] / X_df["age"]
    
    if "trestbps" in X_df.columns and "age" in X_df.columns:
        X_df["bps_per_age"] = X_df["trestbps"] / X_df["age"]
    
    if "thalach" in X_df.columns and "age" in X_df.columns:
        X_df["hr_ratio"] = X_df["thalach"] / X_df["age"]
        
    return X_df