from copy import copy
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
import numpy as np
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

###############################
from statsmodels.tsa.ar_model import AutoReg # Autoregression (AR) model
from statsmodels.tsa.stattools import adfuller, kpss

from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.api import VAR



######################


pd.options.display.float_format = '{:,e}'.format
pd.options.display.width = 0


st.set_page_config(
     page_title = "HARNet | Research Project",
     page_icon=":chart_with_upwards_trend:",
     layout="wide",
     initial_sidebar_state="expanded"
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

variableOption = st.sidebar.radio("Options", ["Univariable", "Multivariable"], index = 0, horizontal = True)

if variableOption == "Univariable":

    variableList = df_symbol_.columns.tolist()
    variable = st.sidebar.selectbox("Variable", index = 18, options = variableList)
    
else:
    nVariables = st.sidebar.number_input("Select number of variables:", min_value=2, max_value=100, value=2, step=1)
    
    
    
    for n in range(nVariables):
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
    
    analysisMethod = st.sidebar.selectbox('Analysis', ['Test', 'Autocorrelation', 'Partial Autocorrelation', 'Trend'])
    
    if analysisMethod == 'Test':
        
        # Check stationarity
        
        if st.sidebar.button('ADF'): # (Reference: https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/)
            
            dftest = adfuller(df_symbol_[variable], autolag='AIC')
            
            st.title('ADF Test Results:')            
            st.write(f"**ADF Statistic**: {dftest[0]}")
            st.write(f"**P-value**: {dftest[1]}")
            st.write(f"**Num of lags**: {dftest[2]}")
            st.write(f"**Num Obs used for ADF | Crit Values Calc**: {dftest[3]}")
            st.write(f"**Critical Values**:")
            for key, value in dftest[4].items():
                st.write(f'{key}, {value}')
                
            if dftest[0] > 0.05:
                st.error("*P-value* is **higher** than the *significance level* of 0.05 and hence there is no reason to **reject the null hypothesis** and take that the series is **NON-stationary**.")
                st.snow()
                
            else:
                st.success("*P-value* is very **less** than the *significance level* of 0.05 and hence we can **reject the null hypothesis** and take that the series is **stationary**.")
                st.balloons()
                
        if st.sidebar.button('KPSS'): # (Reference: https://www.machinelearningplus.com/time-series/kpss-test-for-stationarity/)
            
            statistic, p_value, n_lags, critical_values = kpss(df_symbol_[variable])
            
            st.title('KPSS Test Results:')
            st.write(f"**KPSS Statistic**: {statistic}")
            st.write(f"**P-value**: {p_value}")
            st.write(f"**Num of lags**: {n_lags}")
            st.write(f"**Critical Values**:")
            for key, value in critical_values.items():
                st.write(f'{key}, {value}')
                
            if p_value > 0.05:
                st.error("*P-value* is **higher** than the *significance level* of 0.05 and hence there is no reason to **reject the null hypothesis** and take that the series is **NON-stationary**.")
                st.snow()
                
            else:
                st.success("*P-value* is very **less** than the *significance level* of 0.05 and hence we can **reject the null hypothesis** and take that the series is **stationary**. Now, you can proceed with the **Modeling**.")
                st.balloons()

    
    elif analysisMethod == 'Autocorrelation':         
            

        # Autocorrelation
        lags_ = st.sidebar.number_input("Please enter the # lags:", min_value=1, max_value=100, value=1, step=1)
        data_acf = acf(df_symbol_[variable], nlags=lags_)
        
        acfValues = st.sidebar.radio("Show Autocorrelation coefficient", ["Yes", "No"], index=1, horizontal = True)
        
        fig_acf = plot_acf(df_symbol_[variable], lags = lags_)
        st.pyplot(fig_acf)
        
        if acfValues == 'Yes':
            
            st.write("Autocorrelation coefficient:", data_acf)  
        
    
    elif analysisMethod == 'Partial Autocorrelation':

        # Partial Autocorrelation
        lags_ = st.sidebar.number_input("Please enter the # lags:", min_value=1, max_value=100, value=1, step=1)
        data_pacf = pacf(df_symbol_[variable])
        
        pacfValues = st.sidebar.radio("Show Partial Autocorrelation coefficient", ["Yes", "No"], index=1, horizontal = True)
        
        fig_pacf = plot_pacf(df_symbol_[variable], lags = lags_)
        st.pyplot(fig_pacf)
        
        if pacfValues == 'Yes':
            
            st.write("Partial Autocorrelation coefficient:", data_pacf)  
            
    elif analysisMethod == 'Trend':

        # Seasonal Decompose
        # st.write(df_symbol_)
        # seasonal_df = df_symbol_['rv5_ss']
        # calculate the trend component
        window_ = st.sidebar.slider('Window Size', 1, 365, 1)
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

    st.sidebar.expander("Model") # Reference https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
    model_ = st.sidebar.selectbox("Select Model", ["AR", "MA", "ARMA", "ARIMA", "Prophet", "Neural Prophet", "LSTM", "GRU" , "HARNet"]) # For Multi-variate

    modelExpander = st.sidebar.expander("Parameters")


    if model_ == "AR":
        df_AR = df_symbol_[['DATE', variable]]
        
        def ARModelPlot(dataOriginal, dataForecast):
 
            fig1 = go.Figure()
            
            xAxis1 = dataOriginal['DATE']
            yAxis1 = dataOriginal[variable]
            fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis1, name = 'Original', line=dict(color="#0043ff")))
            
            yAxis2 = dataForecast
            fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis2, name = 'Forecast', line=dict(color="#ee00ff", width=1, dash='dash')))
            
            fig1.layout.update(
                    title_text = "AR Model Prediction",
                    xaxis_rangeslider_visible = True)
        
            st.plotly_chart(fig1)
    

        ARExpander =  st.sidebar.expander("Parameters")
        ARlag_ = ARExpander.slider("Lags", min_value = 1, max_value = 100, step = 1, value = 1)
        testDays_ = ARExpander.slider("Test Days", min_value = 1, max_value = int(len(df_AR)), step = 1, value = 365)
        
        futureDays_ = ARExpander.slider("Forecast Days", min_value = 1, max_value = 3650, step = 1, value = 365)
            
        if st.sidebar.button('Forecast'):
                       
            train = df_AR[:len(df_AR)-testDays_]
            train_ = train.set_index('DATE')
            
            
            test = df_AR[len(df_AR)-testDays_:]
            test_ = test.set_index('DATE')

            train_ = train_.values
            
            model = AutoReg(train_, lags = ARlag_).fit()
            
            # Make predictions of Test Set and compare
            ARpred = model.predict(start = len(train_), end = len(df_AR) - 1, dynamic = False)
            
            test['ARpred'] = ARpred
                       
            # Calculate the error
           
            ARrmse = round(np.sqrt(mean_squared_error(test_[variable], ARpred)), 5)
            ARrmae = round(np.sqrt(mean_absolute_error(test_[variable], ARpred)), 5)
            
            st.sidebar.metric("RMSE", ARrmse)
            st.sidebar.metric("RMAE", ARrmae)
            
            # Forecast
            
            dfTest = []

            dfTest = df_AR['DATE']
            
            
            ts = df_AR['DATE'].max()
            
           
            fdates = ts + pd.Timedelta(days=futureDays_)
            
            fdates_ = pd.DataFrame(pd.date_range(ts, fdates), columns=['future_date'])
                        
            fdates_['DATE'] = fdates_['future_date'].dt.date
            
            fdates_['ARfuture'] = model.predict(start = len(df_AR), end = len(df_AR) + futureDays_, dynamic = False)
            
            fdates_ = fdates_.drop('future_date', axis=1)
            
            # st.write(fdates_)
            
            ################## Plot Results ##################
            
            fig1 = go.Figure()
            
            # Train Data
        
            xAxis1 = train['DATE']
            yAxis1 = train[variable]
            fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis1, name = 'Train', line=dict(color="#0043ff")))
            
            # Test Data
            
            xAxis2 = test['DATE']
            yAxis2 = test[variable]

            fig1.add_trace(go.Scatter( x = xAxis2, y = yAxis2, name = 'Test', line=dict(color="#21ff00")))
            
            # Prediction Data
            
            yAxis3 = test['ARpred']

            fig1.add_trace(go.Scatter( x = xAxis2, y = yAxis3, name = 'Prediction', line=dict(color="#ff0000")))
            
            # Forecast
            
            xAxis4 = fdates_['DATE']
            yAxis4 = fdates_['ARfuture']

            fig1.add_trace(go.Scatter( x = xAxis4, y = yAxis4, name = 'Forecast', line=dict(color="#dc00ff")))
            
            fig1.layout.update(
            title_text = "AR Prediction",
            xaxis_rangeslider_visible = True)
        
            st.plotly_chart(fig1)
    
    elif model_ == "MA":
        df_MA = df_symbol_[['DATE', variable]]
        
        def MAModelPlot(dataOriginal, dataForecast):
 
            fig1 = go.Figure()
            
            xAxis1 = dataOriginal['DATE']
            yAxis1 = dataOriginal[variable]
            fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis1, name = 'Original', line=dict(color="#0043ff")))
            
            yAxis2 = dataForecast
            fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis2, name = 'Forecast', line=dict(color="#ee00ff", width=1, dash='dash')))
            
            fig1.layout.update(
                    title_text = "AR Model Prediction",
                    xaxis_rangeslider_visible = True)
        
            st.plotly_chart(fig1)
    

        MAExpander =  st.sidebar.expander("Parameters")
        MA_q_ = MAExpander.slider("(MA) q", min_value = 1, max_value = 10, step = 1, value = 1)
        testDays_ = MAExpander.slider("Test Days", min_value = 1, max_value = int(len(df_MA)), step = 1, value = 365)
        
        futureDays_ = MAExpander.slider("Forecast Days", min_value = 1, max_value = 3650, step = 1, value = 365)
            
        if st.sidebar.button('Forecast'):
                       
            train = df_MA[:len(df_MA)-testDays_]
            train_ = train.set_index('DATE')
            
            
            test = df_MA[len(df_MA)-testDays_:]
            test_ = test.set_index('DATE')

            train_ = train_.values
            
            model = ARIMA(train_, order=(0, 0, MA_q_)).fit() ################
            
            # Make predictions of Test Set and compare
            MApred = model.predict(start = len(train_), end = len(df_MA) - 1, dynamic = False)
            
            test['MApred'] = MApred
                       
            # Calculate the error
           
            MArmse = round(np.sqrt(mean_squared_error(test_[variable], MApred)), 5)
            MArmae = round(np.sqrt(mean_absolute_error(test_[variable], MApred)), 5)
            
            st.sidebar.metric("RMSE", MArmse)
            st.sidebar.metric("RMAE", MArmae)
            
            # Forecast
            
            dfTest = []

            dfTest = df_MA['DATE']
            
            
            ts = df_MA['DATE'].max()
            
           
            fdates = ts + pd.Timedelta(days=futureDays_)
            
            fdates_ = pd.DataFrame(pd.date_range(ts, fdates), columns=['future_date'])
                        
            fdates_['DATE'] = fdates_['future_date'].dt.date
            
            fdates_['MAfuture'] = model.predict(start = len(df_MA), end = len(df_MA) + futureDays_, dynamic = False)
            
            fdates_ = fdates_.drop('future_date', axis=1)
            
            # st.write(fdates_)
            
            ################## Plot Results ##################
            
            fig1 = go.Figure()
            
            # Train Data
        
            xAxis1 = train['DATE']
            yAxis1 = train[variable]
            fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis1, name = 'Train', line=dict(color="#0043ff")))
            
            # Test Data
            
            xAxis2 = test['DATE']
            yAxis2 = test[variable]

            fig1.add_trace(go.Scatter( x = xAxis2, y = yAxis2, name = 'Test', line=dict(color="#21ff00")))
            
            # Prediction Data
            
            yAxis3 = test['MApred']

            fig1.add_trace(go.Scatter( x = xAxis2, y = yAxis3, name = 'Prediction', line=dict(color="#ff0000")))
            
            # Forecast
            
            xAxis4 = fdates_['DATE']
            yAxis4 = fdates_['MAfuture']

            fig1.add_trace(go.Scatter( x = xAxis4, y = yAxis4, name = 'Forecast', line=dict(color="#dc00ff")))
            
            fig1.layout.update(
            title_text = "MA Prediction",
            xaxis_rangeslider_visible = True)
        
            st.plotly_chart(fig1)
    
    elif model_ == "ARMA":
        df_ARMA = df_symbol_[['DATE', variable]]
        
        def ARMAModelPlot(dataOriginal, dataForecast):
 
            fig1 = go.Figure()
            
            xAxis1 = dataOriginal['DATE']
            yAxis1 = dataOriginal[variable]
            fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis1, name = 'Original', line=dict(color="#0043ff")))
            
            yAxis2 = dataForecast
            fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis2, name = 'Forecast', line=dict(color="#ee00ff", width=1, dash='dash')))
            
            fig1.layout.update(
                    title_text = "ARMA Model Prediction",
                    xaxis_rangeslider_visible = True)
        
            st.plotly_chart(fig1)
    

        ARMAExpander =  st.sidebar.expander("Parameters")
        ARMA_p_ = ARMAExpander.slider("(AR) p", min_value = 1, max_value = 10, step = 1, value = 2)
        ARMA_q_ = ARMAExpander.slider("(MA) q", min_value = 1, max_value = 10, step = 1, value = 1)
        testDays_ = ARMAExpander.slider("Test Days", min_value = 1, max_value = int(len(df_ARMA)), step = 1, value = 365)
        
        futureDays_ = ARMAExpander.slider("Forecast Days", min_value = 1, max_value = 3650, step = 1, value = 365)
            
        if st.sidebar.button('Forecast'):
                       
            train = df_ARMA[:len(df_ARMA)-testDays_]
            train_ = train.set_index('DATE')
            
            
            test = df_ARMA[len(df_ARMA)-testDays_:]
            test_ = test.set_index('DATE')

            train_ = train_.values
            
            model = ARIMA(train_, order=(ARMA_p_, 0, ARMA_q_)).fit() ################
            
            # Make predictions of Test Set and compare
            ARMApred = model.predict(start = len(train_), end = len(df_ARMA) - 1, dynamic = False)
            
            test['ARMApred'] = ARMApred
                       
            # Calculate the error
           
            ARMArmse = round(np.sqrt(mean_squared_error(test_[variable], ARMApred)), 5)
            ARMArmae = round(np.sqrt(mean_absolute_error(test_[variable], ARMApred)), 5)
            
            st.sidebar.metric("RMSE", ARMArmse)
            st.sidebar.metric("RMAE", ARMArmae)
            
            # Forecast
            
            dfTest = []

            dfTest = df_ARMA['DATE']
            
            
            ts = df_ARMA['DATE'].max()
            
           
            fdates = ts + pd.Timedelta(days=futureDays_)
            
            fdates_ = pd.DataFrame(pd.date_range(ts, fdates), columns=['future_date'])
                        
            fdates_['DATE'] = fdates_['future_date'].dt.date
            
            fdates_['ARMAfuture'] = model.predict(start = len(df_ARMA), end = len(df_ARMA) + futureDays_, dynamic = False)
            
            fdates_ = fdates_.drop('future_date', axis=1)
            
            # st.write(fdates_)
            
            ################## Plot Results ##################
            
            fig1 = go.Figure()
            
            # Train Data
        
            xAxis1 = train['DATE']
            yAxis1 = train[variable]
            fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis1, name = 'Train', line=dict(color="#0043ff")))
            
            # Test Data
            
            xAxis2 = test['DATE']
            yAxis2 = test[variable]

            fig1.add_trace(go.Scatter( x = xAxis2, y = yAxis2, name = 'Test', line=dict(color="#21ff00")))
            
            # Prediction Data
            
            yAxis3 = test['ARMApred']

            fig1.add_trace(go.Scatter( x = xAxis2, y = yAxis3, name = 'Prediction', line=dict(color="#ff0000")))
            
            # Forecast
            
            xAxis4 = fdates_['DATE']
            yAxis4 = fdates_['ARMAfuture']

            fig1.add_trace(go.Scatter( x = xAxis4, y = yAxis4, name = 'Forecast', line=dict(color="#dc00ff")))
            
            fig1.layout.update(
            title_text = "ARMA Prediction",
            xaxis_rangeslider_visible = True)
        
            st.plotly_chart(fig1)
    
    elif model_ == "ARIMA":
        df_ARIMA = df_symbol_[['DATE', variable]]
        
        def ARIMAModelPlot(dataOriginal, dataForecast):
 
            fig1 = go.Figure()
            
            xAxis1 = dataOriginal['DATE']
            yAxis1 = dataOriginal[variable]
            fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis1, name = 'Original', line=dict(color="#0043ff")))
            
            yAxis2 = dataForecast
            fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis2, name = 'Forecast', line=dict(color="#ee00ff", width=1, dash='dash')))
            
            fig1.layout.update(
                    title_text = "ARMA Model Prediction",
                    xaxis_rangeslider_visible = True)
        
            st.plotly_chart(fig1)
    

        ARIMAExpander =  st.sidebar.expander("Parameters")
        ARIMA_p_ = ARIMAExpander.slider("(AR) p", min_value = 1, max_value = 10, step = 1, value = 2)
        ARIMA_i_ = ARIMAExpander.slider("(Integration) i", min_value = 1, max_value = 10, step = 1, value = 1)
        ARIMA_q_ = ARIMAExpander.slider("(MA) q", min_value = 1, max_value = 10, step = 1, value = 1)
        testDays_ = ARIMAExpander.slider("Test Days", min_value = 1, max_value = int(len(df_ARIMA)), step = 1, value = 365)
        
        futureDays_ = ARIMAExpander.slider("Forecast Days", min_value = 1, max_value = 3650, step = 1, value = 365)
            
        if st.sidebar.button('Forecast'):
                       
            train = df_ARIMA[:len(df_ARIMA)-testDays_]
            train_ = train.set_index('DATE')
            
            
            test = df_ARIMA[len(df_ARIMA)-testDays_:]
            test_ = test.set_index('DATE')

            train_ = train_.values
            
            model = ARIMA(train_, order=(ARIMA_p_, ARIMA_i_, ARIMA_q_)).fit() ################
            
            # Make predictions of Test Set and compare
            ARIMApred = model.predict(start = len(train_), end = len(df_ARIMA) - 1, dynamic = False)
            
            test['ARIMApred'] = ARIMApred
                       
            # Calculate the error
           
            ARIMArmse = round(np.sqrt(mean_squared_error(test_[variable], ARIMApred)), 5)
            ARIMArmae = round(np.sqrt(mean_absolute_error(test_[variable], ARIMApred)), 5)
            
            st.sidebar.metric("RMSE", ARIMArmse)
            st.sidebar.metric("RMAE", ARIMArmae)
            
            # Forecast
            
            dfTest = []

            dfTest = df_ARIMA['DATE']
            
            
            ts = df_ARIMA['DATE'].max()
            
           
            fdates = ts + pd.Timedelta(days=futureDays_)
            
            fdates_ = pd.DataFrame(pd.date_range(ts, fdates), columns=['future_date'])
                        
            fdates_['DATE'] = fdates_['future_date'].dt.date
            
            fdates_['ARIMAfuture'] = model.predict(start = len(df_ARIMA), end = len(df_ARIMA) + futureDays_, dynamic = False)
            
            fdates_ = fdates_.drop('future_date', axis=1)
            
            # st.write(fdates_)
            
            ################## Plot Results ##################
            
            fig1 = go.Figure()
            
            # Train Data
        
            xAxis1 = train['DATE']
            yAxis1 = train[variable]
            fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis1, name = 'Train', line=dict(color="#0043ff")))
            
            # Test Data
            
            xAxis2 = test['DATE']
            yAxis2 = test[variable]

            fig1.add_trace(go.Scatter( x = xAxis2, y = yAxis2, name = 'Test', line=dict(color="#21ff00")))
            
            # Prediction Data
            
            yAxis3 = test['ARIMApred']

            fig1.add_trace(go.Scatter( x = xAxis2, y = yAxis3, name = 'Prediction', line=dict(color="#ff0000")))
            
            # Forecast
            
            xAxis4 = fdates_['DATE']
            yAxis4 = fdates_['ARIMAfuture']

            fig1.add_trace(go.Scatter( x = xAxis4, y = yAxis4, name = 'Forecast', line=dict(color="#dc00ff")))
            
            fig1.layout.update(
            title_text = "ARIMA Prediction",
            xaxis_rangeslider_visible = True)
        
            st.plotly_chart(fig1)


    elif model_ == "Prophet":
        df_Prophet = df_symbol_[['DATE', variable]]
        
        def ProphetModelPlot(dataOriginal, dataForecast):
 
            fig1 = go.Figure()
            
            xAxis1 = dataOriginal['DATE']
            yAxis1 = dataOriginal[variable]
            fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis1, name = 'Original', line=dict(color="#0043ff")))
            
            yAxis2 = dataForecast
            fig1.add_trace(go.Scatter( x = xAxis1, y = yAxis2, name = 'Forecast', line=dict(color="#ee00ff", width=1, dash='dash')))
            
            fig1.layout.update(
                    title_text = "ARMA Model Prediction",
                    xaxis_rangeslider_visible = True)
        
            st.plotly_chart(fig1)
    

        ProphetExpander =  st.sidebar.expander("Parameters")
        daily_seasonality_ = ProphetExpander.radio("Daily Seasonality", [True, False], index = 1, horizontal = True)
        weekly_seasonality_ = ProphetExpander.radio("Weekly Seasonality", [True, False], index = 1, horizontal = True)
        yearly_seasonality_ = ProphetExpander.radio("Yearly Seasonality", [True, False], index = 1, horizontal = True)

        testDays_ = ProphetExpander.slider("Test Days", min_value = 1, max_value = int(len(df_Prophet)), step = 1, value = 365)
        
        futureDays_ = ProphetExpander.slider("Forecast Days", min_value = 1, max_value = 3650, step = 1, value = 365)
            
        if st.sidebar.button('Forecast'):
                       
            train_ = df_Prophet[:len(df_Prophet)-testDays_]
        
            train_.columns = ['ds', 'y']
            
            # st.write("Train", train_)
            # st.write("length train_:",len(train_))
            
            test = df_Prophet[len(df_Prophet)-testDays_:]
            test_ = test.set_index('DATE')
            # fdates_ = fdates_.drop('future_date', axis=1)
            
            # st.write("length test:", len(test_))
            # st.write("test_", test_)

            model = Prophet()
            model.fit(train_) 
                 
            future=model.make_future_dataframe(periods=futureDays_)
            forecast=model.predict(future)
            
            col1, col2 = st.columns(2)
                
            with col1:
                st.title("Forecast")
                fig1 = plot_plotly(model, forecast)
                fig1.update_layout(
                    autosize=False,
                    width=700,
                    height=400)
                
                st.plotly_chart(fig1) 
                    
            with col2:
                st.title("Components")
                fig2 = plot_components_plotly(model, forecast)
                fig2.update_layout(
                    autosize=False,
                    width=700,
                    height=400)
                
                st.plotly_chart(fig2)                      


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