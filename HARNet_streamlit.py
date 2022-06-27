from requests import options
import streamlit as st
import pandas as pd
from plotly import graph_objects as go
import datetime

pd.options.display.float_format = '{:,e}'.format
pd.options.display.width = 0


st.set_page_config(
     page_title = "HARNet | Research Project",
     page_icon=":chart_with_upwards_trend:",
     layout="wide",
     initial_sidebar_state="expanded"
 )

# ---- Header ----

st.write("HARNet Project")

with st.container():
    st.subheader("University of Vienna")
    st.title("HARNet | Research Project")
    st.write("Reference: [HARNet](https://arxiv.org/abs/1903.04909)")

####

st.sidebar.title("HARNet Menu")

st.sidebar.subheader("Select a dataset")

dataset = st.sidebar.selectbox("Dataset", ["MAN", "OXFORD"]) # , "USEPUINDXD", "VIXCLS"

# Load Data

if dataset == "OXFORD":
    df = pd.read_csv("data/MAN_data_edit.csv") # Read .csv file
    
    symbolsList = df['Symbol'].unique()
    
    symbol = st.sidebar.selectbox("Symbol", index = 5, options = symbolsList)

elif dataset == "MAN":
    df = pd.read_csv("data/MAN_data_edit.csv") # Read .csv file

@st.cache
def plot(df):
    if dataset == "OXFORD":
        fig = go.Figure()
        xAxis = df['DATE']
        yAxis = st.sidebar.selectbox('Select Y-Axis value', index = 15, options = df.columns[1:])
        fig.add_trace(go.Scatter( x = xAxis, y = yAxis))
        fig.layout.update(
            title_text = "Realized Variance Time Series | Dow Jones Industrial Average Index",
            xaxis_rangeslider_visible = True)
        st.plotly_chart(fig)
       
showData = st.sidebar.radio(
     "Show Data Table",
     ('True', 'False'), index = 1, horizontal = True)

d_initial = st.sidebar.date_input(
     "Initial Date",
     datetime.date(2001, 1, 9))

d_final= st.sidebar.date_input(
     "Final Date",
     datetime.date(2017, 12, 17))

if d_initial < d_final:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (d_initial, d_final))
else:
    st.sidebar.error('Error: End date must fall after start date.')

if showData == 'True':
    st.subheader('Data')
    st.dataframe(df)

plot(df)
