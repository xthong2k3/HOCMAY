import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp CSV và thay đổi dấu phẩy thành dấu chấm cho các giá trị số thập phân
data = pd.read_csv('Book1.csv')
# Thay thế các giá trị trống hoặc '-' bằng 0 để xử lý
data.replace(['-', ''], 0, inplace=True)
# Loại bỏ bất kỳ hàng nào chứa giá trị NaN trong cột 'Chỉ số AQI'
data.dropna(subset=['Chỉ số AQI'], inplace=True)

# Chọn các biến đầu vào và biến mục tiêu
X = data[['CO', 'Sương', 'Độ ẩm', 'NO2', 'O3', 'Áp suất', 'Bụi PM10', 'Bụi PM2.5', 'Nhiệt độ', 'Tốc độ gió']]
y = data['Chỉ số AQI']

# Sử dụng K-Means để phân cụm dữ liệu
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(X)

# Tạo mô hình hồi quy tuyến tính cho từng nhóm dữ liệu hoặc toàn bộ dữ liệu
regression_model = LinearRegression()
svm_regressor = SVR(kernel='linear')    

for cluster_id in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster_id]
    X_cluster = cluster_data[['CO', 'Sương', 'Độ ẩm', 'NO2', 'O3', 'Áp suất', 'Bụi PM10', 'Bụi PM2.5', 'Nhiệt độ', 'Tốc độ gió']]
    y_cluster = cluster_data['Chỉ số AQI']
    regression_model.fit(X_cluster, y_cluster)
    svm_regressor.fit(X_cluster, y_cluster)

# Tạo mô hình hồi quy đa biến cho toàn bộ dữ liệu
regression_model.fit(X, y)
svm_regressor.fit(X, y)

# Dự đoán Chỉ số AQI
y_pred = regression_model.predict(X)
y_pred_svm = svm_regressor.predict(X)
# Kết hợp dự đoán từ cả hai mô hình bằng trung bình cộng
y_pred_combined = (y_pred + y_pred_svm) / 2
# Đánh giá mô hình bằng Mean Squared Error, Mean Absolute Error và R-squared
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

mse_svm = mean_squared_error(y, y_pred_svm)
mae_svm = mean_absolute_error(y, y_pred_svm)
r2_svm = r2_score(y, y_pred_svm)

mse_combined = mean_squared_error(y, y_pred_combined)
mae_combined = mean_absolute_error(y, y_pred_combined)
r2_combined = r2_score(y, y_pred_combined)
# In kết quả đánh giá
print("Kết quả đánh giá hồi quy tuyến tính:")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

print("\nKết quả đánh giá SVM:")
print(f"Mean Squared Error: {mse_svm}")
print(f"Mean Absolute Error: {mae_svm}")
print(f"R-squared: {r2_svm}")

print("\nKết quả đánh giá mô hình kết hợp:")
print(f"Mean Squared Error: {mse_combined}")
print(f"Mean Absolute Error: {mae_combined}")
print(f"R-squared: {r2_combined}")

# Vẽ biểu đồ thể hiện kết quả dự đoán từ cả hai mô hình và mô hình kết hợp
plt.scatter(y, y_pred, label='AQI Dự Đoán (Hồi quy tuyến tính)')
plt.scatter(y, y_pred_svm, label='AQI Dự Đoán (SVM)')
plt.scatter(y, y_pred_combined, label='AQI Dự Đoán (Kết hợp)')
plt.xlabel("Chỉ số AQI ban đầu")
plt.ylabel("Chỉ số AQI dự đoán")
plt.title("Biểu đồ kết quả dự đoán và AQI ban đầu")
plt.legend()
plt.show()
# Vẽ biểu đồ phân cụm dữ liệu
for cluster_id in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster_id]
    plt.scatter(cluster_data['CO'], cluster_data['Chỉ số AQI'], label=f'Cluster {cluster_id}')
plt.xlabel("CO")
plt.ylabel("Chỉ số AQI")
plt.title("Biểu đồ phân cụm dữ liệu")
plt.legend()
plt.show()
# Dự đoán chỉ số mới
new_data = pd.DataFrame({'CO': [0.5], 'Sương': [1.0], 'Độ ẩm': [70.0], 'NO2': [15.0], 'O3': [30.0], 'Áp suất': [1000.0], 'Bụi PM10': [60.0], 'Bụi PM2.5': [40.0], 'Nhiệt độ': [25.0], 'Tốc độ gió': [8.0]})
predicted_aqi = regression_model.predict(new_data)
predicted_aqi_svm = svm_regressor.predict(new_data)
predicted_aqi_combined = (predicted_aqi +predicted_aqi_svm ) / 2
print("\nDự đoán Chỉ số AQI cho dữ liệu mới (hồi quy tuyến tính):", predicted_aqi)
print("Dự đoán Chỉ số AQI cho dữ liệu mới (SVM):", predicted_aqi_svm)
print("Dự đoán Chỉ số AQI cho dữ liệu mới (Kết hợp):", predicted_aqi_combined)
# Hiển thị 20 dòng đầu tiên của DataFrame kết quả
results = pd.DataFrame({'AQI Ban Đầu': y, 'AQI Dự Đoán': y_pred, 'AQI Dự Đoán SVM': y_pred_svm, 'AQI Kết hợp':y_pred_combined})
print("\n20 dòng đầu tiên của DataFrame kết quả:")
print(results.head(20))