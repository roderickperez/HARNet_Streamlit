from requests import options
import streamlit as st
import pandas as pd
from plotly import graph_objects as go
import datetime
from plotly.subplots import make_subplots
from typing import List
from dataclasses import field

pd.options.display.float_format = '{:,e}'.format
pd.options.display.width = 0


st.set_page_config(
     page_title = "HARNet | Research Project",
     page_icon=":chart_with_upwards_trend:",
     layout="wide",
     #initial_sidebar_state="expanded"
 )

# ---- Header ----

st.write("HARNet Project")

with st.container():
    st.subheader("University of Vienna")
    st.title("HARNet | Research Project")
    st.write("Reference: [HARNet](https://arxiv.org/abs/1903.04909)")

####

st.sidebar.title("HARNet Menu")

datasetExpander =  st.sidebar.expander("Dataset")

datasetExpander.subheader("Select a dataset")

dataset = datasetExpander.selectbox("Dataset", ["OXFORD"]) # , "USEPUINDXD", "VIXCLS"

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

# Show Data
showData = datasetExpander.radio(
     "Show Data Table",
     ('True', 'False'), index = 1, horizontal = True)
    
if showData == 'True':
    st.subheader('Data')
    st.dataframe(df_symbol_)
    
def plot(data):
    col1, col2 = st.columns(2)
    
    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    xAxis = data['DATE']
    fig.add_trace(go.Scatter( x = xAxis, y = data['rv5_ss'], name = 'Realized Variance (rv5_ss)'))
    fig.add_trace(go.Scatter( x = xAxis, y = data['open_price'], name = 'Open Price'), secondary_y=True)
    fig.add_trace(go.Scatter( x = xAxis, y = data['close_price'], name = 'Close Price'), secondary_y=True)
    fig.layout.update(
        title_text = "Realized Variance Time Series | Dow Jones Industrial Average Index",
        xaxis_rangeslider_visible = True)
    # Set y-axes titles
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Variance", secondary_y=True)
    
    with col1:
        st.plotly_chart(fig)
        
    with col2:
        st.plotly_chart(fig)
       
plot(df_symbol_)

##########################################

st.sidebar.subheader("Parameters")

modelExpander =  st.sidebar.expander("Model")
model_ = modelExpander.selectbox("Select Model", ["HARNet", "Prophet"])
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