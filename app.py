# Generates the home page for the streamlit app. This has all of the css, text and instructions on how to use the app
# This is the only code responsible for the user interphase


# Loading packages to be used
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd

# Page setup
st.set_page_config(page_title="Gone Phishing", layout="wide")

# Custom Header Section
st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background-color: #e8f0fe; border-radius: 10px;'>
        <h1 style='font-size: 3em; margin-bottom: 0.2em;'>Gone Phishing</h1>
        <p style='font-size: 1.2em; color: #333;'>Evaluate suspicious messages!</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Branding
with st.sidebar:
    #st.image("C:/Users/txcwa/txcwalker/static/images/GonePhishingLogo.png", width=120)  # Optional logo
    st.markdown("### Gone Phishing")
    st.markdown("Built by [Cameron Walker](https://www.linkedin.com/in/cameronjwalker9/)")
    st.markdown("[View on GitHub](https://github.com/txcwalker)")
    st.markdown("For questions: txcwalker@gmail.com")

# Introduction
st.markdown("## How It Works")
st.markdown("""
If you're unsure about the legitimacy of a message you've received, simply paste it below and hit **Evaluate**.
This tool uses a custom-trained NLP model to assess whether the message is likely written by a **human** or a **potential bot/script/phishing actor**.

**Important:**  
While this tool is often accurate, it may not reliably distinguish between:
- Benign bots (like auto-responders)
- Malicious scripts or phishing attempts 

This tool is meant to be only part of a holisitic decision making process.
""")

# Load model + pipeline
model_path = "models/checkpoint-58174"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Input form
st.markdown("## Message Analysis")
text = st.text_area("Paste the message you'd like to evaluate:", height=200)

if st.button("Evaluate Message"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred = torch.argmax(probs).item()

    label = "Bot-like / Spam / Phishing" if pred == 1 else " Human/Benign Script"
    confidence = probs[0][pred].item()

    st.markdown(f"### **Prediction:** {label}")
    st.markdown(f"**Confidence Score:** `{confidence:.2%}`")

    # Confidence bar chart
    st.markdown("### Confidence Breakdown")
    conf_df = pd.DataFrame({
        "Label": ["Human-like", "Bot-like/Phishing"],
        "Confidence": [probs[0][0].item(), probs[0][1].item()]
    })
    st.bar_chart(conf_df.set_index("Label"))

# Footer & More Info
st.markdown("---")
st.markdown("""
For a full write-up on the model design, data collection, results and evaluation  
visit [my project post](https://txcwalker.github.io/).
""")
