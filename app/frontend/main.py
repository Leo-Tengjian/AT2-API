import streamlit as st
import requests
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


API_URL = os.getenv("API_URL", "http://localhost:8000")
#API_URL = "http://backend:8000"

def predict_prophet(date):
    response = requests.post(f"{API_URL}/predict/prophet/", json={"date": date})
    if response.status_code == 200:
        return response.json()  
    else:
        raise Exception("Failed", response.status_code, response.text)

st.title('Interactive Data App')

st.write("### Prophet Forecast for the Next 7 Days")
prophet_date = st.text_input("Enter date here")

if st.button('Predict with Prophet'):
    try:
        results = predict_prophet(prophet_date)

        st.write("Forecast Results:")
        forecast_data = []
        for date, prediction in results.items():
            formatted_prediction = f"{prediction:.2f}"
            st.write(f"{date}: {formatted_prediction}")
            forecast_data.append({"date": date, "prediction": prediction})

        df = pd.DataFrame(forecast_data)
        df['date'] = pd.to_datetime(df['date'])

        fig, ax = plt.subplots()
        ax.plot(df['date'], df['prediction'], marker='o', linestyle='-')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  
        plt.xticks(rotation=45)  
        plt.tight_layout()  

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")