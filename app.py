import os
import urllib.request
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN
# ==========================================
st.set_page_config(
    page_title="Quản lý Nề nếp",
    page_icon="🏫",
    layout="centered"
)

st.title("🏫 HỆ THỐNG QUẢN LÝ NỀ NẾP TÁC PHONG")
st.markdown(
    "Ứng dụng AI nhận diện: **Đồng phục, Đầu tóc, Giày dép**."
)

# ==========================================
# 2. TẢI VÀ KHỞI TẠO MÔ HÌNH YOLO
# ==========================================
MODEL_PATH = Path("best_v2.1.pt")
MODEL_URL = (
    "https://github.com/ductrongle76-sudo/"
    "quan-ly-ne-nep/releases/download/v2.1/best.pt"
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        try:
            request = urllib.request.Request(
                MODEL_URL,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urllib.request.urlopen(request, timeout=120) as response:
                with open(MODEL_PATH, "wb") as file:
                    file.write(response.read())

        except Exception as e:
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()

            raise RuntimeError(
                "Không tải được model từ GitHub. "
                "Hãy kiểm tra đường dẫn Release và file best.pt."
            ) from e

    try:
        return YOLO(str(MODEL_PATH))
    except Exception as e:
        raise RuntimeError(
            "Không thể mở model. File best.pt có thể bị lỗi "
            "hoặc không tương thích với phiên bản Ultralytics."
        ) from e


try:
    model = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

# ==========================================
# 3. ĐIỀU CHỈNH NGƯỠNG TIN CẬY
# ==========================================
conf_threshold = st.slider(
    "Ngưỡng tin cậy (Confidence)",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05,
    help="Tăng ngưỡng để lọc các dự đoán có độ tin cậy thấp."
)

# ==========================================
# 4. CHỌN NGUỒN ẢNH
# ==========================================
source = st.radio(
    "Chọn cách kiểm tra:",
    ("Tải ảnh từ máy", "Chụp từ Webcam")
)

if source == "Tải ảnh từ máy":
    upload_img = st.file_uploader(
        "Chọn một bức ảnh",
        type=["jpg", "jpeg", "png"]
    )
else:
    upload_img = st.camera_input("Chụp ảnh từ Webcam")

# ==========================================
# 5. NHẬN DIỆN ẢNH
# ==========================================
if upload_img is not None:
    try:
        image = Image.open(upload_img)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.image(image, caption="Ảnh gốc", width=400)

        if st.button("🔍 Nhận diện vi phạm", type="primary"):
            with st.spinner("AI đang phân tích ảnh..."):
                # PIL cung cấp ảnh RGB.
                # Ultralytics nhận ảnh NumPy theo định dạng HWC BGR
                # khi đầu vào là mảng NumPy.
                img_rgb = np.array(image)
                img_bgr = img_rgb[:, :, ::-1].copy()

                results = model.predict(
                    source=img_bgr,
                    conf=conf_threshold,
                    verbose=False
                )

                result = results[0]

                # Vẽ khung nhận diện, kết quả trả về ảnh BGR.
                result_bgr = result.plot()
                result_rgb = result_bgr[:, :, ::-1].copy()

                st.success("Đã phân tích ảnh!")

                st.image(
                    result_rgb,
                    caption="Kết quả nhận diện của YOLO",
                    width=600
                )

                # Thống kê số đối tượng phát hiện theo lớp.
                if result.boxes is not None and len(result.boxes) > 0:
                    st.subheader("📊 Thống kê nhận diện")

                    counts = {}
                    for cls_id in result.boxes.cls.cpu().tolist():
                        class_name = result.names[int(cls_id)]
                        counts[class_name] = counts.get(class_name, 0) + 1

                    for class_name, count in counts.items():
                        st.write(f"- **{class_name}:** {count}")

                    st.caption(
                        "Các đối tượng được phát hiện không đồng nghĩa "
                        "với việc học sinh chắc chắn vi phạm nội quy. "
                        "Cần đối chiếu với quy định của nhà trường."
                    )
                else:
                    st.info(
                        "Không phát hiện đối tượng nào vượt ngưỡng "
                        "tin cậy đã chọn. Điều này không đảm bảo "
                        "ảnh không có vi phạm."
                    )

    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh: {e}")
