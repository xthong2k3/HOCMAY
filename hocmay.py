import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Đọc dữ liệu từ tệp CSV và thay đổi dấu phẩy thành dấu chấm cho các giá trị số thập phân
data = pd.read_csv('Book1.csv')
# Thay thế các giá trị trống hoặc '-' bằng 0 để xử lý
data.replace(['-', ''], 0, inplace=True)
# Loại bỏ bất kỳ hàng nào chứa giá trị NaN trong cột 'Chỉ số AQI'
data.dropna(subset=['Chỉ số AQI'], inplace=True)

# Chọn các biến đầu vào và biến mục tiêu
X = data[['CO', 'Sương', 'Độ ẩm', 'NO2', 'O3', 'Áp suất', 'Bụi PM10', 'Bụi PM2.5', 'Nhiệt độ', 'Tốc độ gió']]
y = data['Chỉ số AQI']

# Tạo mô hình hồi quy tuyến tính và mô hình SVM
regression_model = LinearRegression()
svm_regressor = SVR(kernel='linear')

# Tạo mô hình hồi quy đa biến cho toàn bộ dữ liệu
regression_model.fit(X, y)
svm_regressor.fit(X, y)

# Dự đoán Chỉ số AQI
y_pred = regression_model.predict(X)
y_pred_svm = svm_regressor.predict(X)

# Kết hợp dự đoán từ cả hai mô hình bằng trung bình cộng
y_pred_combined = (y_pred + y_pred_svm) / 2

# Đánh giá mô hình bằng Mean Squared Error, Mean Absolute Error và R-squared

mse = mean_squared_error(y, y_pred) # hồi quy tuyến tính
mse_svm = mean_squared_error(y, y_pred_svm) # svm
mse_combined = mean_squared_error(y, y_pred_combined) # mô hình kết hợp

mae = mean_absolute_error(y, y_pred) # hồi quy tuyến tính
mae_svm = mean_absolute_error(y, y_pred_svm)#svm
mae_combined = mean_absolute_error(y, y_pred_combined) # mô hình kết hợp

r2 = r2_score(y, y_pred) # hồi quy tuyến tính
r2_svm = r2_score(y, y_pred_svm) # svm
r2_combined = r2_score(y, y_pred_combined) # mô hình kết hợp

# In kết quả đánh giá
print("Kết quả đánh giá MSE:")
print(f"MSE Hồi quy tuyến tính: {mse}")
print(f"MSE SVM: {mse_svm}")
print(f"MSE mô hình kết hợp: {mse_combined}")

print("\nKết quả đánh giá MAE:")
print(f"MAE Hồi quy tuyến tính: {mae}")
print(f"MAE SVM: {mae_svm}")
print(f"MAE mô hình kết hợp: {mae_combined}")

print("\nKết quả đánh giá R-squared:")
print(f"R-squared Hồi quy tuyến tính: {r2}")
print(f"R-squared SVM: {r2_svm}")
print(f"R-squared mô hình kết hợp: {r2_combined}")

# Vẽ biểu đồ thể hiện kết quả dự đoán từ cả hai mô hình và mô hình kết hợp
plt.scatter(y, y_pred, label='AQI Dự Đoán (Hồi quy tuyến tính)')
plt.scatter(y, y_pred_svm, label='AQI Dự Đoán (SVM)')
plt.scatter(y, y_pred_combined, label='AQI Dự Đoán (Kết hợp)')
plt.xlabel("Chỉ số AQI ban đầu")
plt.ylabel("Chỉ số AQI dự đoán")
plt.title("Biểu đồ kết quả dự đoán và AQI ban đầu")
plt.legend()
plt.show()

# Dự đoán chỉ số mới
new_data = pd.DataFrame({'CO': [0.5], 'Sương': [1.0], 'Độ ẩm': [70.0], 'NO2': [15.0], 'O3': [30.0], 'Áp suất': [1000.0], 'Bụi PM10': [60.0], 'Bụi PM2.5': [40.0], 'Nhiệt độ': [25.0], 'Tốc độ gió': [8.0]})
predicted_aqi = regression_model.predict(new_data)
predicted_aqi_svm = svm_regressor.predict(new_data)
predicted_aqi_combined = (predicted_aqi + predicted_aqi_svm) / 2
print("\nDự đoán Chỉ số AQI cho dữ liệu mới (hồi quy tuyến tính):", predicted_aqi)
print("Dự đoán Chỉ số AQI cho dữ liệu mới (SVM):", predicted_aqi_svm)
print("Dự đoán Chỉ số AQI cho dữ liệu mới (Kết hợp):", predicted_aqi_combined)

# Hiển thị 20 dòng đầu tiên của DataFrame kết quả
results = pd.DataFrame({'AQI Ban Đầu': y, 'AQI Dự Đoán': y_pred, 'AQI Dự Đoán SVM': y_pred_svm, 'AQI Kết hợp': y_pred_combined})
print("\n20 dòng đầu tiên của DataFrame kết quả:")
print(results.head(20))

# Chuyển numpy array thành DataFrame
y_pred_df = pd.DataFrame(y_pred, columns=['AQI_DuDoan'])
# Chuyển Chỉ số AQI thực tế và dự đoán thành các nhãn phân loại
# Chia thành 4 nhóm: "Tốt", "Trung bình", "Kém","Xấu"
def convert_to_category(aqi):
    if aqi <= 50:
        return "Tốt"
    elif 50 < aqi <= 100:
        return "Trung bình"
    elif 100< aqi <=150:
        return "Kém"
    else:
        return 'Xấu'
# Áp dụng hàm convert_to_category cho cột AQI_DuDoan
y_pred_categories = y_pred_df['AQI_DuDoan'].apply(convert_to_category)

# Chuyển Chỉ số AQI thực tế thành các nhãn phân loại
y_true_categories = y.apply(convert_to_category)
# Tính độ chính xác
accuracy = accuracy_score(y_true_categories, y_pred_categories)

# In kết quả độ chính xác
print(f"Độ chính xác: {accuracy}")

