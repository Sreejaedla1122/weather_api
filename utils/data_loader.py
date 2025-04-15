import pandas as pd
import re
import os

def load_data(filepath=os.path.join("data","AQI By State 1980-2022.csv")):
    """
    Load and preprocess the AQI dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing AQI data
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with AQI data
    """
    try:
        # Load data
        df = pd.read_csv(filepath)
        
        # Clean column names if needed (just in case)
        df.columns = df.columns.str.strip()
        
        # Extract latitude and longitude from Geo_Loc if needed for additional processing
        # This can be done in specific visualization functions later
        
        # Add any additional preprocessing steps here
        # For example, adding derived metrics
        df['Total_Unhealthy_Days'] = (df['Unhealthy for Sensitive Groups Days'] + 
                                      df['Unhealthy Days'] + 
                                      df['Very Unhealthy Days'] + 
                                      df['Hazardous Days'])
        
        # Calculate percentage of days reported with AQI
        df['Reporting_Rate'] = df['Dys_w_AQI'] / (df['Dys_w_AQI'] + df['Dys_NM']) * 100
        
        # Calculate dominant pollutant
        pollutant_cols = ['Days CO', 'Days NO2', 'Days Ozone', 'Days PM2.5', 'Days PM10']
        df['Dominant_Pollutant'] = df[pollutant_cols].idxmax(axis=1).str.replace('Days ', '')
        
        print(f"Data loaded successfully with {len(df)} records from {df['Year'].min()} to {df['Year'].max()}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return an empty dataframe with same columns to avoid breaking the app
        empty_df = pd.DataFrame(columns=[
            'Geo_Loc', 'Year', 'State', 'Pop_Est', 'TTL_Cnty', 'Cnty_Rpt',
            'Dys_w_AQI', 'Dys_NM', 'Dys_Blw_Thr', 'Dys_Abv_Thr', 
            'Good Days', 'Moderate Days', 'Unhealthy for Sensitive Groups Days',
            'Unhealthy Days', 'Very Unhealthy Days', 'Hazardous Days',
            'Max AQI', '90th Percentile AQI', 'Median AQI',
            'Days CO', 'Days NO2', 'Days Ozone', 'Days PM2.5', 'Days PM10',
            'Total_Unhealthy_Days', 'Reporting_Rate', 'Dominant_Pollutant'
        ])
        return empty_df

def extract_coordinates(geo_str):
    """
    Extract longitude and latitude from Geo_Loc string
    
    Parameters:
    -----------
    geo_str : str
        String containing geographic coordinates in the format "POINT (lon lat)"
        
    Returns:
    --------
    tuple
        (longitude, latitude)
    """
    pattern = r"POINT \(([-\d.]+) ([-\d.]+)\)"
    match = re.search(pattern, geo_str)
    if match:
        lon, lat = match.groups()
        return float(lon), float(lat)
    return None, None

def get_state_summary(df, state=None, year=None):
    """
    Generate summary statistics for a specific state and/or year
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing AQI data
    state : str, optional
        State to filter data for
    year : int, optional
        Year to filter data for
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics
    """
    filtered_df = df.copy()
    
    if state:
        filtered_df = filtered_df[filtered_df['State'] == state]
    
    if year:
        filtered_df = filtered_df[filtered_df['Year'] == year]
    
    # Calculate summary statistics
    summary = {
        'Median AQI (avg)': filtered_df['Median AQI'].mean(),
        'Max AQI (avg)': filtered_df['Max AQI'].mean(),
        'Good Days (avg)': filtered_df['Good Days'].mean(),
        'Unhealthy Days (avg)': filtered_df['Unhealthy Days'].mean(),
        'Days with PM2.5 (avg)': filtered_df['Days PM2.5'].mean(),
        'Days with Ozone (avg)': filtered_df['Days Ozone'].mean()
    }
    
    return pd.DataFrame(summary, index=[0])
