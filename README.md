![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-purple)

Hệ thống dự đoán nguy cơ mắc bệnh tim mạch dựa trên các chỉ số y tế lâm sàng. Dự án sử dụng kỹ thuật **Machine Learning (Stacking Ensemble)** kết hợp với **Feature Engineering** để đạt độ chính xác cao nhất, được triển khai dưới dạng Web API với giao diện người dùng thân thiện.

---

## 📑 Mục lục
- [Giới thiệu](#giới-thiệu)
- [Cơ sở Lý thuyết & Phương pháp](#cơ-sở-lý-thuyết--phương-pháp)
  - [1. Bộ dữ liệu](#1-bộ-dữ-liệu)
  - [2. Feature Engineering (Kỹ thuật đặc trưng)](#2-feature-engineering-kỹ-thuật-đặc-trưng)
  - [3. Mô hình Stacking Ensemble](#3-mô-hình-stacking-ensemble)
- [Cấu trúc Dự án](#cấu-trúc-dự-án)
- [Cài đặt & Hướng dẫn sử dụng](#cài-đặt--hướng-dẫn-sử-dụng)
- [API Documentation](#api-documentation)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)

---

## Giới thiệu
[cite_start]Bệnh tim mạch là nguyên nhân gây tử vong hàng đầu thế giới. Việc chẩn đoán sớm đóng vai trò quan trọng trong điều trị. Dự án này xây dựng một hệ thống hỗ trợ ra quyết định (CDSS) giúp các bác sĩ hoặc người dùng cá nhân đánh giá nhanh nguy cơ dựa trên các thông số như tuổi, cholesterol, huyết áp, v.v.

## Cơ sở Lý thuyết & Phương pháp

### 1. Bộ dữ liệu
[cite_start]Dự án sử dụng bộ dữ liệu **Cleveland Heart Disease** từ UCI Machine Learning Repository.
- **Kích thước:** 303 bản ghi.
- [cite_start]**Đặc trưng (Features):** 13 đặc trưng lâm sàng (Tuổi, Giới tính, CP, Trestbps, Chol, FBS, Restecg, Thalach, Exang, Oldpeak, Slope, CA, Thal).
- **Nhãn (Target):** 0 (Không bệnh) và 1 (Có bệnh).

### 2. Feature Engineering (Kỹ thuật đặc trưng)
[cite_start]Thay vì chỉ sử dụng dữ liệu thô, dự án áp dụng kỹ thuật Feature Engineering để tạo ra các đặc trưng mới nhằm làm nổi bật tín hiệu và cải thiện khả năng học của mô hình. Các đặc trưng mới được tạo ra bao gồm:

* **Cholesterol per Age (`chol_per_age`):** Tỷ lệ Cholesterol trên tuổi. [cite_start]Phản ánh mức độ tích tụ mỡ máu tương đối theo độ lão hóa[cite: 424].
* [cite_start]**Blood Pressure per Age (`bps_per_age`):** Tỷ lệ Huyết áp tâm thu trên tuổi.
* [cite_start]**Heart Rate Ratio (`hr_ratio`):** Tỷ lệ Nhịp tim tối đa trên tuổi.
* [cite_start]**Age Bining:** Phân nhóm độ tuổi để xử lý tốt hơn các xu hướng phi tuyến tính.

[cite_start]Kết quả thực nghiệm cho thấy việc áp dụng Feature Engineering giúp tăng độ chính xác đáng kể so với dữ liệu gốc (từ ~84% lên ~90-93% trên tập test).

### 3. Mô hình Stacking Ensemble
Để tối ưu hóa hiệu suất, dự án sử dụng kỹ thuật **Ensemble Learning** dạng **Stacking**. [cite_start]Đây là phương pháp kết hợp sức mạnh của nhiều mô hình cơ sở để giảm sai số và tăng độ ổn định.
Kiến trúc mô hình bao gồm:
1.  **Level-0 (Base Learners):**
    * **K-Nearest Neighbors (KNN):** Dựa trên khoảng cách giữa các điểm dữ liệu. [cite_start]K tối ưu được chọn thông qua Cross-Validation (~11).
    * [cite_start]**Decision Tree (DT):** Mô hình cây quyết định với độ sâu giới hạn để tránh overfitting.
    * [cite_start]**Naive Bayes (NB):** Dựa trên định lý Bayes với giả định các đặc trưng độc lập.
2.  **Level-1 (Meta Learner):**
    * [cite_start]Sử dụng **KNN** để tổng hợp kết quả dự đoán (xác suất) từ các mô hình Level-0 và đưa ra kết quả cuối cùng.

---

## Cấu trúc Dự án
```bash
heart_disease_project/
├── data/
│   └── cleveland.csv      # Dữ liệu gốc
├── model/
│   └── heart_model.pkl    # Pipeline mô hình đã huấn luyện (bao gồm cả xử lý dữ liệu)
├── main.py                # Backend API (FastAPI)
├── train.py               # Script huấn luyện mô hình & Feature Engineering
├── index.html             # Giao diện người dùng (Frontend)
├── requirements.txt       # Danh sách thư viện
└── README.md              # Tài liệu dự án
````

-----

## Cài đặt & Hướng dẫn sử dụng

### Bước 1: Clone và cài đặt môi trường

```bash
# Clone dự án (nếu có git)
git clone <your-repo-url>
cd heart_disease_project

# Tạo môi trường ảo (Khuyên dùng)
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### Bước 2: Huấn luyện mô hình

Trước khi chạy ứng dụng, bạn cần huấn luyện mô hình để tạo file `heart_model.pkl`. Quá trình này bao gồm cả bước tiền xử lý và Feature Engineering tự động.

```bash
python train.py
```

*Output mong đợi: `Độ chính xác trên tập test: 0.9xxx` và thông báo đã lưu model.*

### Bước 3: Khởi chạy Server

```bash
python main.py
```

Server sẽ khởi động tại `http://127.0.0.1:8000`.

### Bước 4: Sử dụng Giao diện

Mở file `index.html` bằng trình duyệt bất kỳ (Chrome, Firefox, Edge). Nhập các chỉ số sức khỏe và nhấn **"Dự đoán ngay"**.

-----

## API Documentation

### Endpoint: `/predict`

  * **Method:** `POST`
  * **Description:** Nhận dữ liệu lâm sàng và trả về dự đoán nguy cơ bệnh tim cùng các chỉ số phân tích.

**Request Body (JSON):**

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

**Response (JSON):**

```json
{
  "prediction": 1,
  "result_text": "Có nguy cơ bệnh tim",
  "confidence": 85.5,
  "features_engineering": {
    "chol_per_age": 3.698,
    "bps_per_age": 2.301,
    "hr_ratio": 2.381
  }
}
```

-----

## Công nghệ sử dụng

  * **Ngôn ngữ:** Python 3.9+
  * **Data Processing:** Pandas, NumPy
  * **Machine Learning:** Scikit-learn (Pipeline, StackingClassifier, Imputer)
  * **Backend:** FastAPI, Uvicorn
  * **Frontend:** HTML5, Bootstrap 5, JavaScript (Fetch API)

-----


<!-- end list -->

