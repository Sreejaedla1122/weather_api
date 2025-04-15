# AQI Dashboard - Multi-page Streamlit Application

This is an interactive dashboard application that visualizes Air Quality Index (AQI) data across the United States from 1980 to 2022.

## Features

- **Interactive Visualizations**: Dynamic plots using Plotly that respond to user inputs
- **Geospatial Analysis**: Map-based visualizations using state coordinates
- **Time Series Analysis**: Trend analysis across multiple years
- **State-by-State Comparison**: Compare AQI metrics between different states
- **Pollutant Analysis**: Detailed breakdown of specific air pollutants

## Project Structure

```
aqi_dashboard/
├── app.py                  # Main application entry point
├── utils/                  # Utility functions
│   ├── data_loader.py      # Data loading and preprocessing
│   └── visualization.py    # Common visualization functions
└── README.md               # Documentation
└── requirements.txt        # project requirements
```

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
streamlit run app.py
```

## Data Description

The dataset contains AQI metrics for U.S. states from 1980 to 2022, including:

- Geographic location data
- Population estimates
- County reporting statistics
- Various AQI categories (Good, Moderate, Unhealthy, etc.)
- Maximum, 90th percentile, and median AQI values
- Days where specific pollutants were the main AQI contributors (CO, NO2, Ozone, PM2.5, PM10)

