import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
st.write(cv2.__file__)
st.write(getattr(cv2, "__version__", "no version"))
st.write([n for n in dir(cv2) if "Cascade" in n])

# ---------------------------------
# 1. Streamlit Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Haar Cascade Face Detection",
    page_icon="👤",
    layout="wide"
)

# ---------------------------------
# 2. Application Titlepip install opencv-python
# ---------------------------------

st.title("👤 Haar Cascade Face Detection")

st.write(
    "Upload an image to detect human faces "
    "using OpenCV Haar Cascade Classifier."
)

st.divider()

# ---------------------------------
# 3. Load Haar Cascade Classifier
# ---------------------------------
cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    st.error("Error: Could not load Haar Cascade XML file.")
    st.stop()

# ---------------------------------
# 4. Upload Image
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# 5. Face Detection
# ---------------------------------

if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Convert image to NumPy array
    img_array = np.array(image)

    # Convert RGB image to BGR for OpenCV
    img_bgr = cv2.cvtColor(
        img_array,
        cv2.COLOR_RGB2BGR
    )

    # Convert image into grayscale
    gray = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Copy original image
    output_image = img_array.copy()

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:

        cv2.rectangle(
            output_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            output_image,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    # ---------------------------------
    # 6. Display Images
    # ---------------------------------

    st.subheader("Detection Results")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            img_array,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            output_image,
            caption="Face Detection Output",
            use_container_width=True
        )

    # ---------------------------------
    # 7. Display Face Count
    # ---------------------------------

    st.divider()

    st.subheader("Detection Information")

    st.metric(
        label="Total Faces Detected",
        value=len(faces)
    )

    if len(faces) > 0:
        st.success(
            f"Successfully detected {len(faces)} face(s)!"
        )
    else:
        st.warning("No faces detected in the uploaded image.")

    # ---------------------------------
    # 8. Download Output Image
    # ---------------------------------

    result_image = Image.fromarray(output_image)

    from io import BytesIO

    buffer = BytesIO()

    result_image.save(buffer, format="PNG")

    st.download_button(
        label="Download Detected Image",
        data=buffer.getvalue(),
        file_name="face_detection_result.png",
        mime="image/png"
    )

else:
    st.info("Please upload an image to start face detection.")
