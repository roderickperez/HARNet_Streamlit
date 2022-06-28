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
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


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
    st.write("##### Heterogeneous Autoregressive Model of Realized Volatility | Reference: [HARNet](https://arxiv.org/abs/2205.07719)")

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

variableList = df_symbol_.columns.tolist()
variable = st.sidebar.selectbox("Variable", index = 18, options = variableList)

################################
# Horizontal Menu
selected = option_menu(
    menu_title = None,
    options = ["Data", "Stats", "Pre-Process", "Plot", "Analysis", "Forecast"],
    icons = ['table', 'clipboard-data', 'sliders', 'graph-up', 'activity', 'share'],
    orientation = "horizontal",
    default_index = 3,
)
   
def plot(data):
    col1, col2 = st.columns(2)
    
    fig1 = go.Figure()
    # fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    xAxis = data['DATE']
    fig1.add_trace(go.Scatter( x = xAxis, y = data[variable], name = 'Selected Variable'))

    fig1.layout.update(
        title_text = "Selected Variable",
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
        


if selected == 'Data':
    st.subheader('Data')
    st.dataframe(df_symbol_)
       
elif selected == 'Stats':
    st.subheader('Stats')
    st.write("Symbol: ", symbol)
    st.dataframe(df_symbol_.describe())

elif selected == 'Pre-Process':
    
    def postProcessPlot(dataOriginal):
 
        fig1 = go.Figure()
        
        xAxis1 = dataOriginal['DATE']
        yAxis1 = dataOriginal[variable]
        fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis1, name = 'Original', line=dict(color="#0043ff")))
        
        df_preProcess_temp_ = df_preProcess_temp.dropna()
        yAxis2 = df_preProcess_temp_
        fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis2, name = 'Pre Processed ', line=dict(color="#21ff00")))
    
        st.plotly_chart(fig1)
    
    
    st.sidebar.subheader("Pre-Process Parameters")
    
    preProcessMethod = st.sidebar.selectbox('Methods', ['Moving Average', 'Moving Median', 'Moving Standard Deviation'])
    
    if preProcessMethod == 'Moving Average':
        window_ = st.sidebar.slider('Window Size', 1, 100, 1)
        df_preProcess_temp = df_symbol_[variable].rolling(window=window_).mean()

        
        if st.sidebar.button('Add to dataframe'):
            df_symbol_['rv5_ss_MovAve'] = df_preProcess_temp
            st.sidebar.success('Data has been added to the dataframe successfully.')
            
        postProcessPlot(df_symbol_)
        

elif selected == 'Plot':
    plot(df_symbol_)
    
elif selected == 'Analysis':
    
    analysisMethod = st.sidebar.selectbox('Analysis', ['Autocorrelation', 'Partial Autocorrelation', 'Trend'])
    
    if analysisMethod == 'Autocorrelation':

        # Autocorrelation
        lags_ = st.sidebar.number_input("Please enter the order of autocorrelation:", min_value=1, max_value=100, value=1, step=1)
        data_acf = acf(df_symbol_[variable], nlags=lags_)
        
        acfValues = st.sidebar.radio("Show Autocorrelation coefficient", ["Yes", "No"], index=1, horizontal = True)
        
        fig_acf = plot_acf(df_symbol_[variable])
        st.pyplot(fig_acf)
        
        if acfValues == 'Yes':
            
            st.write("Autocorrelation coefficient:", data_acf)  
        
    
    elif analysisMethod == 'Partial Autocorrelation':

        # Partial Autocorrelation
        lags_ = st.sidebar.number_input("Please enter the order of autocorrelation:", min_value=1, max_value=100, value=1, step=1)
        data_pacf = pacf(df_symbol_[variable])
        
        pacfValues = st.sidebar.radio("Show Partial Autocorrelation coefficient", ["Yes", "No"], index=1, horizontal = True)
        
        fig_pacf = plot_pacf(df_symbol_[variable])
        st.pyplot(fig_pacf)
        
        if pacfValues == 'Yes':
            
            st.write("Partial Autocorrelation coefficient:", data_pacf)  
            
    elif analysisMethod == 'Trend':

        # Seasonal Decompose
        # st.write(df_symbol_)
        # seasonal_df = df_symbol_['rv5_ss']
        # calculate the trend component
        window_ = st.sidebar.slider('Window Size', 1, 100, 1)
        center_ = st.sidebar.radio('Center', [True, False], index=0, horizontal = True)
        df_symbol_["trend"] = df_symbol_[variable].rolling(window=window_, center=True).mean()

        # detrend the series
        df_symbol_["detrended"] = df_symbol_[variable] - df_symbol_["trend"]

        # calculate the seasonal component
        df_symbol_.index = pd.to_datetime(df_symbol_.index)
        df_symbol_["month"] = df_symbol_.index.month
        df_symbol_["seasonality"] = df_symbol_.groupby("month")["detrended"].transform("mean")

        # get the residuals
        df_symbol_["resid"] = df_symbol_["detrended"] - df_symbol_["seasonality"]
        
        # st.write(df_symbol_)


        def seasonalityPlot(dataOriginal):
 
            fig1 = go.Figure()
            
            fig1.add_trace(go.Scatter( x = dataOriginal['DATE'], y = dataOriginal[variable], name = 'Original'))
            fig1.add_trace(go.Scatter( x = dataOriginal['DATE'], y = dataOriginal['trend'], name = 'Trend'))
            fig1.add_trace(go.Scatter( x = dataOriginal['DATE'], y = dataOriginal['detrended'], name = 'Detrended'))
            fig1.add_trace(go.Scatter( x = dataOriginal['DATE'], y = dataOriginal['month'], name = 'Month'))
            fig1.add_trace(go.Scatter( x = dataOriginal['DATE'], y = dataOriginal['seasonality'], name = 'Seasonality'))
            fig1.add_trace(go.Scatter( x = dataOriginal['DATE'], y = dataOriginal['resid'], name = 'Residual'))
            
            fig1.layout.update(xaxis_rangeslider_visible = True)
        
        
            st.plotly_chart(fig1)
                      
        seasonalityPlot(df_symbol_)

else:
    st.sidebar.subheader(" Forecast Parameters")

    st.sidebar.expander("Model")
    model_ = st.sidebar.selectbox("Select Model", ["LSTM", "Prophet", "AR", "MA", "ARMA", "ARIMA", "SARIMA", "SARIMAX", "VAR", "VARMA", "VARMAX", "SES", "HWES", "HARNet"])

    modelExpander = st.sidebar.expander("Parameters")


    if model_ == "LSTM":
        pass


    elif model_ == "Prophet":
        # Keep only two columns
        df_prophet = df_symbol_[['DATE', variable]]
        # Rename columns
        df_prophet.columns = ['ds', 'y']
        
    
        # Phophet Model
        prophetExpander =  st.sidebar.expander("Parameters")
        interval_width_ = prophetExpander.slider("Interval Width (%)", min_value = 0.0, max_value = 1.0, step = 0.1, value = 0.95)
        daily_seasonality_ = prophetExpander.radio("Daily Seasonality", [True, False], index = 1, horizontal = True)
        weekly_seasonality_ = prophetExpander.radio("Weekly Seasonality", [True, False], index = 1, horizontal = True)
        yearly_seasonality_ = prophetExpander.radio("Yearly Seasonality", [True, False], index = 1, horizontal = True)
        
        if st.sidebar.button('Forecast'):
            # Create Model
            m = Prophet(interval_width = interval_width_, daily_seasonality = daily_seasonality_, weekly_seasonality = weekly_seasonality_, yearly_seasonality = yearly_seasonality_)
            model = m.fit(df_prophet)

            # Forecast
            
            periods_ = prophetExpander.slider("Periods (days)", min_value = 1, max_value = 3650, step = 1, value = 365)
            prophetFuture = m.make_future_dataframe(periods = periods_)
            prophetForecast = m.predict(prophetFuture)
            
            showProphetForecast_ = st.sidebar.radio("Show Forecast", [True, False], index = 1, horizontal = True)
            
            if showProphetForecast_ == True:
                st.subheader("Forecast")
                st.dataframe(prophetForecast)
                
            # Save Forecast into new df
            prophetForecast_ = prophetForecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
            def postProphetFocast(dataOriginal, dataForecast):
        
                fig1 = go.Figure()
                # fig1 = make_subplots(specs=[[{"secondary_y": True}]])
                
                xAxis1 = dataOriginal['DATE']
                yAxis1 = dataOriginal[variable]
                fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis1, name = 'Original', line=dict(color="#0043ff", width=1)))
                
                xAxis2 = dataForecast['ds']
                yAxis2 = dataForecast['yhat']
                fig1.add_trace(go.Scatter( x = xAxis2, y = yAxis2, name = 'Forecast', line=dict(color="#ee00ff", width=1, dash='dash')))
                
                fig1.update_layout(
                    autosize = False,
                    width = 1300,
                    height = 500)
                
                fig1.layout.update(
                    title_text = "Forecast",
                    xaxis_rangeslider_visible = True)
                
                st.plotly_chart(fig1)
                
            postProphetFocast(df_symbol_, prophetForecast_)
        



    elif model_ == "HARNet":
        filter_conv_ = modelExpander.slider("Filter Convolution", 1, 10, 1)
        bias_ = modelExpander.radio("Bias", ("True", "False"), index = 1, horizontal = True)
        activation_deconv_ = modelExpander.selectbox("Activation Function", ["relu", "sigmoid", "tanh"], index = 0)

        optimizationExpander =  st.sidebar.expander("Optimization")
        learning_rate_ = optimizationExpander.slider("Learning Rate (10e3)", min_value = 0.0001, max_value = 1.0, step = 0.0001, value = 0.0001)
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