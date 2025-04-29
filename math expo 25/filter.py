import cv2
import numpy as np
import streamlit as st
from scipy import signal
from PIL import Image

# Step 1: Upload Image or Capture from Webcam
st.title("Interactive Filter Design Tool")
st.markdown("### Step 1: Upload Image or Take a Photo")
img_source = st.radio("Choose image source:", ("Upload", "Webcam"))

if img_source == "Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        frame = np.array(img)
elif img_source == "Webcam":
    run = st.button("Take Photo")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    frame = None
    if run:
        ret, frame = camera.read()
        camera.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

# Step 2: Filter Design using Bilinear Transformation
st.markdown("### Step 2: Design Filter")
filter_type = st.selectbox("Choose filter type", ["Low-pass", "High-pass", "Band-pass"])
cutoff = st.slider("Cutoff frequency (Hz)", 1, 100, 10)
sampling_rate = st.slider("Sampling rate (Hz)", 100, 1000, 500)

if filter_type == "Low-pass":
    b, a = signal.butter(4, cutoff / (0.5 * sampling_rate), btype='low', analog=False)
elif filter_type == "High-pass":
    b, a = signal.butter(4, cutoff / (0.5 * sampling_rate), btype='high', analog=False)
elif filter_type == "Band-pass":
    low = cutoff / (0.5 * sampling_rate)
    high = (cutoff + 10) / (0.5 * sampling_rate)
    b, a = signal.butter(4, [low, high], btype='band', analog=False)

# Step 3: Apply Filter (for demonstration, we apply it to grayscale image rows)
st.markdown("### Step 3: Apply Filter")
if 'frame' in locals() and frame is not None:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    filtered_image = np.apply_along_axis(lambda m: signal.lfilter(b, a, m), axis=1, arr=gray)
    st.image(filtered_image, caption="Filtered Image", use_column_width=True)

# Step 4: Save the Output
st.markdown("### Step 4: Save Filtered Image")
if 'filtered_image' in locals():
    save_btn = st.button("Save Image")
    if save_btn:
        img_out = Image.fromarray(np.uint8(filtered_image))
        img_out.save("filtered_output.png")
        st.success("Image saved as filtered_output.png")