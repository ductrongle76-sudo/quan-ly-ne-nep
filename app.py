import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import urllib.request

# 1. Cấu hình giao diện web
st.set_page_config(page_title="Quản lý Nề nếp", page_icon="🏫")
st.title("🏫 HỆ THỐNG QUẢN LÝ NỀ NẾP TÁC PHONG")
st.markdown("Ứng dụng AI nhận diện: **Đồng phục, Đầu tóc, Giày dép**.")

# 2. Load mô hình YOLO11 (tự động tải file best.pt từ GitHub nếu chưa có)
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        # Đây chính là đường link bạn vừa copy
        url = "https://github.com/ductrongle76-sudo/quan-ly-ne-nep/releases/download/v1.0/best.pt"
        urllib.request.urlretrieve(url, model_path)
    return YOLO(model_path)

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
            img_array = np.array(image)
            if img_array.shape[-1] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
            results = model.predict(source=img_array, conf=conf_threshold)
            
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st.success("Hoàn tất phân tích!")
            st.image(res_rgb, caption="Kết quả nhận diện", width=600)