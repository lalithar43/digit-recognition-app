import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="DigitSketch",
    page_icon="✨",
    layout="centered"
)

# =========================================================
# CUSTOM CSS (Optimized)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at top left, rgba(124,58,237,0.2), transparent 30%),
                linear-gradient(135deg, #020617 0%, #0f172a 100%);
    color: white;
}

.main-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 30px;
    padding: 40px;
    backdrop-filter: blur(20px);
}

.title { text-align: center; font-size: 58px; font-weight: 800; margin-bottom: 10px; }
.subtitle { text-align: center; color: #94a3b8; font-size: 18px; margin-bottom: 40px; }

/* Center the canvas drawing area */
.stCanvas { margin: 0 auto; border-radius: 20px; overflow: hidden; }

.stButton > button {
    width: 100%; height: 58px; border-radius: 18px;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white; font-weight: 700; transition: 0.3s;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_digit_model():
    # If you saved weights only, keep your current architecture.
    # If you saved the full model, use: return keras.models.load_model("digit_model.h5")
    model = keras.models.Sequential([
        keras.layers.Input(shape=(28, 28, 1)), # Added channel dimension
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    try:
        model.load_weights("digit_model.weights.h5")
    except:
        st.error("Model weights not found. Please ensure 'digit_model.weights.h5' is in the directory.")
    return model

model = load_digit_model()

# =========================================================
# IMAGE PROCESSING
# =========================================================
def preprocess_image(img_data):
    # Convert RGBA (from canvas) to Grayscale
    img = Image.fromarray(img_data.astype('uint8')).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    
    # IMPORTANT: MNIST is white digits on black background.
    # Our canvas is black ink on white background. We MUST invert.
    img_array = 255 - img_array 
    
    # Normalize
    img_array = img_array.astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28)
    return img_array

# =========================================================
# UI LAYOUT
# =========================================================
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="title">DigitSketch</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Draw a handwritten digit (0-9) below.</div>', unsafe_allow_html=True)

# Canvas Setup
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=20, # Increased for better recognition
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

st.markdown("<br>", unsafe_allow_html=True)

# Predict Logic
_, btn_col, _ = st.columns([1, 1.5, 1])
with btn_col:
    predict_btn = st.button("✨ Predict Digit")

if predict_btn:
    if canvas_result.image_data is not None:
        # Check if the user actually drew something
        if np.unique(canvas_result.image_data).size > 1:
            processed = preprocess_image(canvas_result.image_data)
            prediction = model.predict(processed)
            digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.divider()
            res_c1, res_c2 = st.columns(2)
            res_c1.metric("Predicted Digit", f"{digit}")
            res_c2.metric("Confidence", f"{confidence:.1f}%")
        else:
            st.warning("Please draw a digit first!")

st.markdown('</div>', unsafe_allow_html=True)