from operator import index
from requests import options
import streamlit as st
import pandas as pd
from plotly import graph_objects as go
import datetime
from plotly.subplots import make_subplots
from typing import List
from dataclasses import field
import streamlit_option_menu
from streamlit_option_menu import option_menu
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from prophet import Prophet

pd.options.display.float_format = '{:,e}'.format
pd.options.display.width = 0


st.set_page_config(
     page_title = "HARNet | Research Project",
     page_icon=":chart_with_upwards_trend:",
     layout="wide",
     #initial_sidebar_state="expanded"
 )

# ---- Header ----

with st.container():
    st.title("HARNet")
    st.write("##### Heterogeneous Autoregressive Model of Realized Volatility | Reference: [HARNet](https://arxiv.org/abs/1903.04909)")

####

st.sidebar.title("HARNet Menu")

datasetExpander =  st.sidebar.expander("Dataset")

datasetExpander.subheader("Select a dataset")

dataset = datasetExpander.selectbox("Dataset", ["OXFORD"])

# Load Data

df = pd.read_csv("data/oxfordmanrealizedvolatilityindices.csv") # Read .csv file
# df = df.rename(columns ={'Unnamed: 0':'DATE'})
df['DATE'] = pd.to_datetime(df['Unnamed: 0'],utc=True).dt.date

first_column = df.pop('DATE')

df.insert(0, 'DATE', first_column)
df = df.drop(['Unnamed: 0'], axis=1)

symbolsList = df['Symbol'].unique()

symbol = datasetExpander.selectbox("Symbol", index = 6, options = symbolsList)

# group the data by stock symbol

df_symbol = df.groupby(['Symbol'])
df_symbol_ = df_symbol.get_group(symbol)

d_initial = datasetExpander.date_input(
     "Initial Date",
     datetime.date(2001, 1, 9))

d_final= datasetExpander.date_input(
     "Final Date",
     datetime.date(2017, 12, 17))

if d_initial > d_final:
    datasetExpander.error('Error: End date must fall after start date.')

##### Select subgroup between dates
df_symbol_ = df_symbol_.loc[(df_symbol_['DATE'] > d_initial) & (df_symbol_['DATE'] <= d_final)]


################################
# Horizontal Menu
selected = option_menu(
    menu_title = None,
    options = ["Data", "Stats", "Pre-Process", "Plot", "Trend", "Forecast"],
    icons = ['table', 'clipboard-data', 'sliders', 'graph-up', 'activity', 'share'],
    orientation = "horizontal",
    default_index = 3,
)
   
def plot(data):
    col1, col2 = st.columns(2)
    
    fig1 = go.Figure()
    # fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    xAxis = data['DATE']
    fig1.add_trace(go.Scatter( x = xAxis, y = data['rv5_ss'], name = 'Realized Variance (rv5_ss)'))

    fig1.layout.update(
        title_text = "Realized Variance",
        xaxis_rangeslider_visible = True)
    
    with col1:
        st.plotly_chart(fig1)
        
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter( x = xAxis, y = data['open_price'], name = 'Open Price'))
    fig2.add_trace(go.Scatter( x = xAxis, y = data['close_price'], name = 'Close Price'))
    
    fig2.layout.update(
        title_text = "Open | Close Price",
        xaxis_rangeslider_visible = True)
    
    with col2:
        st.plotly_chart(fig2)
        
def postProcessPlot(data):
 
    fig1 = go.Figure()
    # fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    xAxis1 = df_symbol_['DATE']
    yAxis1 = df_symbol_['rv5_ss']
    fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis1, name = 'Original', line=dict(color="#0043ff")))
    
    df_preProcess_temp_ = df_preProcess_temp.dropna()
    yAxis2 = df_preProcess_temp_
    fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis2, name = 'Pre Processed ', line=dict(color="#21ff00")))
   
    st.plotly_chart(fig1)
    

if selected == 'Data':
    st.subheader('Data')
    st.dataframe(df_symbol_)
       
elif selected == 'Stats':
    st.subheader('Stats')
    st.write("Symbol: ", symbol)
    st.write(df_symbol_.describe())

elif selected == 'Pre-Process':
    col1, col2 = st.columns([2,4])
    
    with col1:
        st.subheader('Parameters')
        preProcessMethod = st.selectbox('Pre-Process Methods', ['Moving Average', 'Moving Median', 'Moving Standard Deviation'])
        
        if preProcessMethod == 'Moving Average':
            window_ = st.slider('Window Size', 1, 10, 1)
            df_preProcess_temp = df_symbol_['rv5_ss'].rolling(window=window_).mean()
            
            if st.button('Add to dataframe'):
                df_symbol_['rv5_ss_MovAve'] = df_preProcess_temp
                st.success('Data has been added to the dataframe successfully.')
            
    with col2:
        postProcessPlot(df_preProcess_temp)
        
        
    
    
elif selected == 'Plot':
    plot(df_symbol_)
    
elif selected == 'Trend':
    pass

else:
    pass


##########################################

st.sidebar.subheader("Parameters")

st.sidebar.expander("Model")
model_ = st.sidebar.selectbox("Select Model", ["LSTM", "Prophet", "AR", "MA", "ARMA", "ARIMA", "SARIMA", "SARIMAX", "VAR", "VARMA", "VARMAX", "SES", "HWES", "HARNet"])

modelExpander = st.sidebar.expander("Parameters")


if model_ == "LSTM":
    pass


elif model_ == "Prophet":
    pass



elif model_ == "HARNet":
    filter_conv_ = modelExpander.slider("Filter Convolution", 1, 10, 1)
    bias_ = modelExpander.radio("Bias", ("True", "False"), index = 1)
    activation_deconv_ = modelExpander.selectbox("Activation Function", ["relu", "sigmoid", "tanh"], index = 0)

    optimizationExpander =  st.sidebar.expander("Optimization")
    learning_rate_ = optimizationExpander.slider("Learning Rate (10e3)", 1, 1000, 1)
    epochs_ = optimizationExpander.slider("Epochs", 10, 100000, 10)
    steps_per_epoch_ = optimizationExpander.slider("Steps per Epochs", 1, 10, 1)
    labelLength_ = optimizationExpander.slider("Label Length", 1, 10, 1)
    batch_size_ = optimizationExpander.slider("Batch Size", 1, 10, 1)
    optimizer_ = optimizationExpander.selectbox("Optimizer", ["Adam", "RMSprop", "SGD"], index = 0)

    preProcessingExpander =  st.sidebar.expander("Pre-Processing Parameters")
    scaler_ = preProcessingExpander.selectbox("Scaler", ["MinMax"], index = 0)


    # UTILS
    class HARNetCfg:
        # Model
        model: str = model_
        filters_dconv: int = int(filter_conv_)
        use_bias_dconv: bool = bias_
        activation_dconv: str = activation_deconv_
        lags: List[int] = field(default_factory=lambda: [1, 5, 20])

        # Optimization
        learning_rate: float = (learning_rate_)/1000
        epochs: int = epochs_
        steps_per_epoch: int = steps_per_epoch_
        label_length: int = labelLength_
        batch_size: int = batch_size_
        optimizer: str = optimizer_
        loss: str = "QLIKE"
        verbose: int = 1
        baseline_fit: str = "WLS"

        # Data
        # path_MAN: str = "./data/MAN_data.csv"
        # asset: str = ".DJI"
        # include_sv: bool = False
        # start_year_train: int = 2012
        # n_years_train: int = 4
        # start_year_test: int = 2016
        # n_years_test: int = 1

        # Preprocessing
        scaler: str = scaler_
        scaler_min: float = 0.0
        scaler_max: float = 0.001

        # Save Paths
        tb_path: str = "./tb/"
        save_path: str = "./results/"
        save_best_weights: bool = False

        # Misc
        run_eagerly: bool = False
    
with st.sidebar.container():
    st.sidebar.subheader("University of Vienna | Research Project")
    st.sidebar.write("###### App Authors: Roderick Perez & Le Thi (Janie) Thuy Trang")
    st.sidebar.write("###### Faculty Advisor: Xandro Bayer")
    st.sidebar.write("###### Updated: 28/6/2022")