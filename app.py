import streamlit as tf_st
import numpy as np
import tensorflow as tf
from PIL import Image

# ==============================================================================
# ULTRA PRO PLATINUM GRAPHICS & LUXURY STYLING (Custom CSS Integration)
# ==============================================================================
tf_st.set_page_config(
    page_title="Skin Cancer Diagnostic Center",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom injection of dynamic glassmorphic premium interface
tf_st.markdown("""
<style>
    /* Premium background grid */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Global Card Designs */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 12px 28px !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s ease-in-out !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(168, 85, 247, 0.6) !important;
    }
    
    /* Header layouts */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        color: #94a3b8;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }
    
    /* Status boxes custom treatment */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MACHINE LEARNING ENGINE ARCHITECTURE
# ==============================================================================
@tf_st.cache_resource
def load_trained_cnn_model():
    try:
        # Tries loading the generated keras framework weights locally
        return tf.keras.models.load_model('skin_cancer_model.h5')
    except Exception as network_error:
        return None

diagnostic_model = load_trained_cnn_model()

# ==============================================================================
# CORE WORKFLOW APPLICATION VIEW
# ==============================================================================
tf_st.markdown('<h1 class="main-title">🔬 AI Skin Lesion Classifier</h1>', unsafe_allow_html=True)
tf_st.markdown('<p class="sub-title">Platinum Core Engine - Multi-channel Smart Diagnostics Pipeline</p>', unsafe_allow_html=True)

# Structural Layout Separation
sidebar_panel = tf_st.sidebar
sidebar_panel.image("https://flaticon.com", width=100)
sidebar_panel.markdown("### 🛠️ Configuration Console")
input_channel = sidebar_panel.radio(
    "Choose Input Stream Source:",
    ("📸 Live HD Camera Capture", "📁 Upload Image Local Storage")
)

sidebar_panel.info("💡 Ensure the physical labels match the expected format before uploading onto production pipelines.")

target_media_file = None

if "Live HD Camera Capture" in input_channel:
    # Activate Streamlit frontend camera interface pipeline 
    target_media_file = tf_st.camera_input("Position the skin lesion target carefully inside the frame focus:")
else:
    # Standard multi-format image loader configuration
    target_media_file = tf_st.file_uploader(
        "Drop diagnostic file into target sector:", 
        type=["jpg", "jpeg", "png", "webp"]
    )

# ==============================================================================
# PIPELINE INFERENCE EXECUTION BLOCK
# ==============================================================================
col1, col2 = tf_st.columns([1, 1])

if target_media_file is not None:
    with col1:
        tf_st.markdown("### 🖼️ Extracted Matrix Source")
        processed_pil_image = Image.open(target_media_file).convert("RGB")
        tf_st.image(processed_pil_image, use_column_width=True, caption="Active Matrix Input View")
        
    with col2:
        tf_st.markdown("### 📊 Diagnostic Evaluation Analytics")
        
        if diagnostic_model is None:
            tf_st.error("🚨 Configuration Error: `skin_cancer_model.h5` path weight target was not located in root repo structures.")
            tf_st.warning("Please ensure to double-check the physical label and check that your model file is saved in the exact same directory.")
        else:
            with tf_st.spinner("⚡ Running Deep CNN Model Inferences Across Target Layers..."):
                # Preprocessing alignment transformation logic (Matches the Kaggle 128x128 array setup)
                resized_image = processed_pil_image.resize((128, 128))
                normalized_array = np.array(resized_image) / 255.0
                expanded_input_batch = np.expand_dims(normalized_array, axis=0)
                
                # Perform continuous predictions
                calculated_raw_probabilities = diagnostic_model.predict(expanded_input_batch)
                malignant_probability_score = float(calculated_raw_probabilities[0][0])
                
                # Class resolution decision layers
                if malignant_probability_score > 0.5:
                    resolved_class_label = "Malignant"
                    confidence_percentage = malignant_probability_score * 100
                    alert_color_theme = "🚨"
                else:
                    resolved_class_label = "Benign"
                    confidence_percentage = (1 - malignant_probability_score) * 100
                    alert_color_theme = "✅"
                
                # Render results inside dynamic visual containers
                tf_st.markdown(f"""
                <div class="metric-container">
                    <h2 style='margin:0; color:#f8fafc;'>Diagnostic Diagnosis:</h2>
                    <h1 style='color:{"#f43f5e" if resolved_class_label == "Malignant" else "#10b981"}; font-size:3.5rem; margin:10px 0;'>
                        {alert_color_theme} {resolved_class_label}
                    </h1>
                    <p style='color:#94a3b8; font-size:1.1rem; margin:0;'>
                        Model Confidence Integrity Level: <strong>{confidence_percentage:.2f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display structural progression metric indicators
                tf_st.write("")
                tf_st.progress(confidence_percentage / 100)
                
                tf_st.caption("⚠️ Verification Notice: This algorithm serves purely informational purposes. Validate details with clinical professionals.")
