import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import unicodedata
import numpy as np
import logging
import time
import json
from datetime import date
import pandas as pd

# FUNCTION TO LOAD MODEL
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("french_gender_predictor.h5", compile= False)
    
    with open("gender_tokenizer.pkl","rb") as f:
        tokenizer = pickle.load(f)
        
    return model, tokenizer

# FUNCTION TO PREDICT GENDER
def predict_gender(name, model, tokenizer, max_length):
    name_clean = name.lower().strip()
    seq = tokenizer.texts_to_sequences([name_clean])
    padded = pad_sequences(seq, maxlen = max_length, padding = "post")
    prob_female = model.predict(padded, verbose = 0)[0][0]
    
    # convert numpy dtype to float
    prob_female = float(prob_female)
    
    if prob_female > 0.5:
        gender = "Female"
        confidence = prob_female
    else:
        gender = "Male"
        confidence = 1 - prob_female
        
    return gender, confidence

# FUNCTION TO LOAD TREND DATA (GROUPED DATA INSEE BY FIRST NAME AND GENDER)
@st.cache_data
def load_trend_data():
    return pd.read_csv("trend_data.csv")

model, tokenizer = load_assets()
max_length = 20
trend_data = load_trend_data()

# STREAMLIT UI 
st.title("French Name Gender Predictor")

name_input = st.text_input("First name:", placeholder = "e.g., Marie")

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

if name_input:
    start_time = time.time()
    
    # STEP 1: PREPROCESS 
    name_clean = unicodedata.normalize("NFC", name_input.lower().strip())
    seq = tokenizer.texts_to_sequences([name_clean])
    padded = pad_sequences(seq, maxlen=18, padding="post")
    
    # STEP 2: PREDICT
    gender_prediction, confidence = predict_gender(name_clean, model, tokenizer, max_length)
    duration_time = time.time() - start_time
    
    log_data = {
        "name": name_input,
        "predicted_gender": gender_prediction,
        "confidence": confidence,
        "latency": duration_time,
        "model_version": "v1.0"
    } 
    
    logger.info(json.dumps(log_data, ensure_ascii=False))
    
    # STEP 3: OUTPUT
    st.success(f"Prediction: {gender_prediction}")
    st.metric(
        label = "Probability",
        value = f"{round(confidence*100, 1)}"
    )
    
    name_stats = trend_data[trend_data["first_name_clean"] == name_clean]
    
    if not name_stats.empty:
        name_stats["gender_label"] = name_stats["gender"].map({1: "Male", 2: "Female"})
        
        st.subheader(f"Historical gender split for {name_input.capitalize()}")
        
        chart_data = name_stats.set_index("gender_label")["count"]
        st.bar_chart(chart_data)
        
        total_births = name_stats["count"].sum()
        stats_dict = {row['gender_label']: (row["count"] / total_births) * 100 for _, row in name_stats.iterrows()}
        
        if len(stats_dict) > 1:
            summary_text = (f"According to INSEE data, {name_input.capitalize()} is a unisex name: "
                            f"{round(stats_dict.get("Female",0),1)}% Female and "
                            f"{round(stats_dict.get("Male",0),1)}% Male."
                            )
        else:
            gender = list(stats_dict.keys())[0]
            summary_text = f"According to INSEE data, 100% of people named {name_input.capitalize()} are {gender}."
            
        st.info(summary_text)    