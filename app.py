# Save as app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io

st.title("MASK DETECTION with YOLO11 – Direct Model Load")

# 1️⃣ Load YOLO11 model directly (from local file)
model = YOLO("best.pt")  # Make sure best.pt is in the same folder
st.success("Loaded YOLO11 model successfully!")

# 2️⃣ Upload images
uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Open image
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        # YOLO inference
        results = model(img_array)

        # Annotate image manually
        annotated_image = img_array.copy()
        boxes = results[0].boxes  # YOLO11 latest API
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color=(0,255,0), thickness=3)
                label = f"LP {conf:.2f}"
                cv2.putText(annotated_image, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Convert annotated image to PIL for Streamlit
        annotated_pil = Image.fromarray(annotated_image)
        st.image(annotated_pil, caption=f"Processed: {uploaded_file.name}", use_column_width=True)

        # Download button
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="Download Highlighted Image",
            data=byte_im,
            file_name=f"highlighted_{uploaded_file.name}",
            mime="image/png"
        )
