# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats.mstats import winsorize
import streamlit as st

# Loading dan data preparation 
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("goldstock_v2.csv")
    df = df.iloc[:, 1:7]
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.rename(columns={"Close/Last": "Close"})
    df['Volume'] = winsorize(df['Volume'], limits=[0, 0.04])
    return df.sort_values('Date')

# Melatih model linear regression
@st.cache_data
def train_model(df):
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # Scaling data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model linear Regression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Mengukur performa model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mse, r2, X_test, y_test, y_pred

# Tab 1 - Prediksi Closing Price baru
def predict_tab(model, scaler):
    st.header("Predict Closing Price :money_with_wings:")

    # Input data
    open_price = st.number_input("Open Price", min_value=0.0)
    high_price = st.number_input("High Price", min_value=0.0)
    low_price = st.number_input("Low Price", min_value=0.0)
    volume = st.number_input("Volume", min_value=0.0)
    
    # Memproses data yang di-input
    if st.button("Predict"):
        new_data = pd.DataFrame([[open_price, high_price, low_price, volume]], columns=['Open', 'High', 'Low', 'Volume'])
        new_data_scaled = scaler.transform(new_data)
        predicted_price = model.predict(new_data_scaled)
        st.subheader(f"Predicted Closing Price :heavy_dollar_sign: : {predicted_price[0]:.2f}")

# Tab 2 - Model Performance dan Plotting linear Regression
def performance_tab(mse, r2, y_test, y_pred):

    # Performa Model
    st.header("Model Performance :chart_with_upwards_trend:")
    col1, col2= st.columns(2)
    with col1:
        st.subheader(f"Mean Squared Error: {mse}")
    with col2:
        st.subheader(f"R-squared Score: {r2}")
    
    st.divider()

    # Plotting 
    st.header("Actual vs Predicted Prices Plot :bar_chart:")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Prices')
    ax.set_ylabel('Predicted Prices')
    ax.set_title('Actual vs Predicted Closing Prices')
    st.pyplot(fig)

# Main function 
def main():
    st.image("./trading_emas.jpg")
    st.title("Gold Price Prediction :moneybag:")

    # Proses data dan training model
    df = load_and_preprocess_data()
    model, scaler, mse, r2, X_test, y_test, y_pred = train_model(df)
    
    # Membuat tab dashboard
    tab1, tab2 = st.tabs(["Predict Closing Price", "Model Performance and Plot"])
    
    # Tab 1
    with tab1:
        predict_tab(model, scaler)
    
    # Tab 2
    with tab2:
        performance_tab(mse, r2, y_test, y_pred)

    css = '''
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:20px;
        }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

# Memanggil main function
if __name__ == "__main__":
    main()
