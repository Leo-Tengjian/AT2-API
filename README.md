# Interactive Data Forecasting App

The application is containerized using Docker, allowing easy deployment and consistent environments.

## Features

- **Prophet Model Forecast**: Enter a date to receive forecasted sales revenue for the next 7 days.
- **
- **Interactive Charts**: Visualize the forecast data in an interactive line chart.
- **Backend API**: FastAPI serves prediction requests through a `/predict/prophet/` endpoint.

## Project Structure
.
├── app
│   ├── backend
│   │   ├── main.py          # FastAPI backend code
│   │   ├── dockerfile        
│   ├── frontend
│   │   ├── main.py          # Streamlit frontend code
│   └── ——— dockerfile 
│   
├── docker-compose.yml        # Docker Compose configuration
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── render.txt
└── models                    # Pretrained models

`date`: date from which the model will predict the forecasted sales for the following 7 days. 
        The expected date format is YYYY-MM-DD.

`store_id`: identifier of the store from which the model will predict the sales on.
`item_id`:  identifier of the item from which the model will predict the sales on.

This project can be deployed to Render