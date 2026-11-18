
import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Face Detection App",
    page_icon="😊",
    layout="centered"
)

st.title("Face Detection Application")
st.write("Upload an image to detect faces.")

# Load Haar Cascade safely
@st.cache_resource
def load_face_detector():
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError(
            f"Could not load Haar Cascade: {cascade_path}"
        )

    return face_cascade


try:
    face_cascade = load_face_detector()
except Exception as e:
    st.error(f"Face detector loading failed: {e}")
    st.stop()


# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    # Convert RGB to grayscale
    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(
            image_array,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

    st.image(
        image_array,
        caption="Face Detection Result",
        use_container_width=True
    )

    st.success(f"Total faces detected: {len(faces)}")
