import streamlit as st
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np
import base64
import json
import os
import pandas as pd
from streamlit_lottie import st_lottie

# --------------- Page Config ------------------
st.set_page_config(page_title="Fake News Detector", layout="centered", page_icon="📰")

# --------------- Set Background ------------------
def set_bg_image(bg_image_path):
    with open(bg_image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg_image("news_bg.jpg")

# --------------- Styling ------------------
bg_color = "rgba(0, 0, 0, 0)"
text_color = "white"
textarea_bg = "#0e0000"
st.markdown(f"""
    <style>
    .main > div {{
        background-color: {bg_color};
        padding: 2rem;
        border-radius: 12px;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
    }}
    h1, h3, p, div, label {{
        color: {text_color} !important;
    }}
    textarea {{
        background-color: {textarea_bg} !important;
        color: {text_color} !important;
        border-radius: 10px;
    }}
    button {{
        border-radius: 10px;
    }}
    </style>
""", unsafe_allow_html=True)


# --------------- Lottie Loader ------------------
def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)


# --------------- Title ------------------
st.title("📰 Fake News Detector")
st.markdown("### Detect whether a news article is *Fake* or *Real*.")
st.markdown("Paste your news content below and click **Detect Fake News** to analyze it.")

# --------------- Load Model ------------------
@st.cache_resource
def load_model():
    model_dir = "bert_fakenews_model"
    model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

model, tokenizer = load_model()

# --------------- Prediction Function ------------------
def predict(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
    return np.argmax(probs), probs

# --------------- Text Input ------------------
user_input = st.text_area("📝 Enter the news article text below:", height=200, placeholder="Paste or type news article content here...")

# --------------- History Init ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------- Detect Button ------------------
if st.button("🔍 Detect Fake News"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            label, probs = predict(user_input)
            fake_prob, real_prob = probs[0], probs[1]

            # Save to history
            st.session_state.history.append({
                "Text": user_input,
                "Prediction": "Fake" if label == 0 else "Real",
                "Confidence (Fake)": round(fake_prob, 2),
                "Confidence (Real)": round(real_prob, 2)
            })

            st.write(f"**Prediction Probabilities**\n\n- Fake → `{fake_prob:.2f}`\n- Real → `{real_prob:.2f}`")

            if fake_prob > 0.6:
                st_lottie(load_lottie("error.json"), height=200)
                st.markdown(f"""
                    <div style='background-color: #ff4d4d; padding: 20px; border-radius: 10px; text-align: center; color: white; font-weight: bold;'>
                        🚫 This news is <span style='font-size: 24px;'>FAKE</span><br>Confidence: {fake_prob:.2f}
                    </div>
                """, unsafe_allow_html=True)
            elif real_prob > 0.6:
                st_lottie(load_lottie("success.json"), height=200)
                st.markdown(f"""
                    <div style='background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center; color: white; font-weight: bold;'>
                        ✅ This news is <span style='font-size: 24px;'>REAL</span><br>Confidence: {real_prob:.2f}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st_lottie(load_lottie("neutral.json"), height=200)
                st.markdown(f"""
                    <div style='background-color: #ffa500; padding: 20px; border-radius: 10px; text-align: center; color: white; font-weight: bold;'>
                        🤔 Model is unsure<br>Fake: {fake_prob:.2f} | Real: {real_prob:.2f}
                    </div>
                """, unsafe_allow_html=True)

# --------------- History Section ------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕓 Prediction History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download Results (CSV)", data=csv, file_name="prediction_results.csv", mime="text/csv")

# --------------- Footer ------------------
st.markdown("---")
st.caption("Made with ❤️ using BERT and Streamlit")
st.markdown("**SHYAM KUMAR SONI**")
