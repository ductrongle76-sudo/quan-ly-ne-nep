import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# 1. Cấu hình giao diện web
st.set_page_config(page_title="Quản lý Nề nếp", page_icon="🏫")
st.title("🏫 HỆ THỐNG QUẢN LÝ NỀ NẾP TÁC PHONG")
st.markdown("Ứng dụng AI nhận diện: **Đồng phục, Đầu tóc, Giày dép**.")

# 2. Load mô hình YOLO11 (chạy 1 lần cho mượt)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 3. Tạo thanh trượt chỉnh độ nhạy
conf_threshold = st.slider("Ngưỡng tin cậy (Confidence)", 0.1, 1.0, 0.25, 0.05)

# 4. Chọn cách tải ảnh
source = st.radio("Chọn cách kiểm tra:", ("Tải ảnh từ máy", "Chụp từ Webcam"))

upload_img = None
if source == "Tải ảnh từ máy":
    upload_img = st.file_uploader("Chọn một bức ảnh...", type=['jpg', 'jpeg', 'png'])
else:
    upload_img = st.camera_input("Chụp ảnh từ Webcam")

# 5. Xử lý khi có ảnh tải lên
if upload_img is not None:
    image = Image.open(upload_img)
    st.image(image, caption="Ảnh gốc", width=400)

    if st.button("🔍 Nhận diện vi phạm"):
        with st.spinner('AI đang phân tích...'):
            # Chuyển ảnh sang định dạng YOLO đọc được
            img_array = np.array(image)
            if img_array.shape[-1] == 4: # Nếu là ảnh đuôi PNG có nền trong suốt
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

            # Chạy AI
            results = model.predict(source=img_array, conf=conf_threshold)

            # Vẽ khung kết quả
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            st.success("Hoàn tất phân tích!")
            st.image(res_rgb, caption="Kết quả nhận diện", width=600)