import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io

st.set_page_config(page_title="Colorful Filter Studio", layout="centered")
st.title("🎨 Colorful Filter Studio")
st.markdown("Upload an image or take one with your webcam and apply cool filters!")

# Step 1: Upload or Capture Image
st.markdown("## 📸 Step 1: Choose an Image")
img_source = st.radio("Select Image Source:", ("Upload", "Webcam"))

frame = None

if img_source == "Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.session_state['captured_image'] = img.copy()
        frame = st.session_state['captured_image']

elif img_source == "Webcam":
    if "captured_image" not in st.session_state:
        st.session_state['captured_image'] = None

    if st.button("Capture Photo"):
        camera = cv2.VideoCapture(0)
        ret, img_cv = camera.read()
        camera.release()
        if ret:
            frame = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            st.session_state['captured_image'] = frame

    if st.session_state['captured_image'] is not None:
        frame = st.session_state['captured_image']

# Step 2: Filter Selection
if frame is not None:
    st.markdown("## 🎨 Step 2: Choose a Filter")
    filter_choice = st.selectbox("Select a Filter", ["None", "Vintage", "Gingham", "Blur", "Black & White", "Lark"])

    def apply_filter(image, filter_name):
        if filter_name == "None":
            return image
        elif filter_name == "Vintage":
            img_np = np.array(image)
            vintage = cv2.applyColorMap(img_np, cv2.COLORMAP_PINK)
            return Image.fromarray(vintage)
        elif filter_name == "Gingham":
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.8)
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(1.1)
        elif filter_name == "Blur":
            return image.filter(ImageFilter.GaussianBlur(5))
        elif filter_name == "Black & White":
            return image.convert("L").convert("RGB")
        elif filter_name == "Lark":
            img_np = np.array(image).astype(np.float32)
            img_np[..., 0] *= 1.1  # Boost red
            img_np[..., 1] *= 1.05  # Boost green
            img_np = np.clip(img_np, 0, 255)
            return Image.fromarray(np.uint8(img_np))

    filtered_img = apply_filter(frame, filter_choice)

    # Step 3: Show filtered image
    st.markdown("## 🖼 Step 3: Preview")
    st.image(filtered_img, caption="Filtered Image", use_column_width=True)

    # Step 4: Save / Download
    st.markdown("## 💾 Step 4: Save or Download")
    buf = io.BytesIO()
    filtered_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Image",
        data=byte_im,
        file_name="filtered_image.png",
        mime="image/png"
    )

    st.success("You can now save the image to your gallery!")

else:
    st.info("Please upload or capture an image to proceed.")
