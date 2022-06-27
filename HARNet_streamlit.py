import streamlit as st
import pandas as pd
import numpy as np
#import tensorflow as tf

#DTYPE = tf.float32
pd.options.display.float_format = '{:,e}'.format
pd.options.display.width = 0

st.set_page_config(page_title = "HARNet | Research Project", 
                  page_icon = ":chart_with_upwards_trend:",
                  layout = "wide")

# ---- Header ----
with st.container():
    st.subheader("University of Vienna")
    st.title("HARNet | Research Project")
    st.write("Reference: [HARNet](https://arxiv.org/abs/1903.04909)")
    
#st.cache

st.sidebar.title("HARNet Menu")

st.sidebar.subheader("Select a dataset")

dataset = st.sidebar.selectbox("Dataset", ["MAN", "USEPUINDXD", "VIXCLS"])

if dataset == "MAN":
    df = pd.read_csv("data/MAN_data.csv")
    df = df.rename(columns ={'Unnamed: 0':'DATE'}) #Rename first column to DATE
    

    
elif dataset == "USEPUINDXD":
    df = pd.read_csv("data/USEPUINDXD.csv")

else:
    df = pd.read_csv("data/VIXCLS.csv")


df_time = pd.to_datetime(df['DATE'],utc=True).dt.date
df.DATE = df_time

pd_data = df.copy()

if dataset == "USEPUINDXD":
    pd_data['VIXCLS'] = pd.to_numeric(pd_data['VIXCLS'],errors = 'coerce')


st.title(dataset)  # add a title
st.write(df)  # visualize my dataframe in the Streamlit app