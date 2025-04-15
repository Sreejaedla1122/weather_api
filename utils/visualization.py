import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import streamlit as st

def create_yearly_aqi_heatmap(df):
    """Creates a heatmap showing AQI metrics over years"""
    yearly_data = df.groupby('Year').agg({
        'Median AQI': 'mean',
        'Max AQI': 'mean',
        '90th Percentile AQI': 'mean'
    }).reset_index()
    
    # Convert the data to a format suitable for a heatmap
    heatmap_data = yearly_data.melt(
        id_vars=['Year'],
        value_vars=['Median AQI', 'Max AQI', '90th Percentile AQI'],
        var_name='Metric',
        value_name='Value'
    )
    
    # Create the heatmap
    fig = px.density_heatmap(
        heatmap_data,
        x='Year',
        y='Metric',
        z='Value',
        color_continuous_scale='Viridis',
        title='AQI Metrics Heatmap Over Years'
    )
    
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='AQI Metric',
        coloraxis_colorbar_title='AQI Value'
    )
    
    return fig

def create_aqi_distribution(df, year):
    """Creates histogram distributions of AQI values for selected year"""
    year_data = df[df['Year'] == year]
    
    fig = px.histogram(
        year_data,
        x='Median AQI',
        color='State',
        marginal='box',
        title=f'Distribution of Median AQI Values by State ({year})',
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis_title='Median AQI',
        yaxis_title='Count',
        legend_title='State'
    )
    
    return fig

def create_pollutant_radar_chart(df, states, year):
    """Creates a radar chart comparing pollutants across selected states"""
    filtered_df = df[(df['State'].isin(states)) & (df['Year'] == year)]
    
    # Group by state and calculate mean for each pollutant
    pollutant_cols = ['Days CO', 'Days NO2', 'Days Ozone', 'Days PM2.5', 'Days PM10']
    state_pollutants = filtered_df.groupby('State')[pollutant_cols].mean().reset_index()
    
    # Create radar chart
    fig = go.Figure()
    
    for i, state in enumerate(state_pollutants['State']):
        fig.add_trace(go.Scatterpolar(
            r=state_pollutants.loc[i, pollutant_cols].values,
            theta=[col.replace('Days ', '') for col in pollutant_cols],
            fill='toself',
            name=state
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, state_pollutants[pollutant_cols].max().max() * 1.1]
            )
        ),
        title=f'Pollutant Comparison by State ({year})',
        showlegend=True
    )
    
    return fig

def create_aqi_forecast(df, state, metric='Median AQI', periods=5):
    """Creates a simple time series forecast for AQI metrics"""
    state_data = df[df['State'] == state].sort_values('Year')
    
    # Use simple moving average for forecasting
    X = state_data['Year'].values
    y = state_data[metric].values
    
    # Simple polynomial fit for forecasting
    degree = 2
    coeffs = np.polyfit(X, y, degree)
    p = np.poly1d(coeffs)
    
    # Create forecast years and values
    last_year = state_data['Year'].max()
    forecast_years = np.array(range(last_year + 1, last_year + periods + 1))
    forecast_values = p(forecast_years)
    
    # Create the plot
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=X,
        y=y,
        mode='markers+lines',
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_years,
        y=forecast_values,
        mode='markers+lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'{metric} Forecast for {state}',
        xaxis_title='Year',
        yaxis_title=metric,
        legend_title='Data Type'
    )
    
    return fig

def create_seasonal_analysis(df, state):
    """Creates a visualization for seasonal patterns in AQI data"""
    # For this demo, we'll simulate seasonal data since the dataset doesn't have month information
    # In a real implementation, you'd need data with month/season information
    
    # Create synthetic seasonal data for demonstration
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    years = sorted(df[df['State'] == state]['Year'].unique())[-10:]  # Last 10 years
    
    # Create synthetic seasonal AQI values (would be replaced with real data)
    seasonal_data = []
    
    for year in years:
        base_aqi = df[(df['State'] == state) & (df['Year'] == year)]['Median AQI'].mean()
        
        # Simulate seasonal variations
        seasonal_data.append({'Year': year, 'Season': 'Winter', 'AQI': base_aqi * 1.2})
        seasonal_data.append({'Year': year, 'Season': 'Spring', 'AQI': base_aqi * 0.9})
        seasonal_data.append({'Year': year, 'Season': 'Summer', 'AQI': base_aqi * 1.3})
        seasonal_data.append({'Year': year, 'Season': 'Fall', 'AQI': base_aqi * 1.0})
    
    seasonal_df = pd.DataFrame(seasonal_data)
    
    fig = px.box(
        seasonal_df,
        x='Season',
        y='AQI',
        color='Season',
        title=f'Seasonal AQI Patterns for {state} (Simulated Data)',
        category_orders={'Season': seasons}
    )
    
    fig.update_layout(
        xaxis_title='Season',
        yaxis_title='Median AQI',
        showlegend=False
    )
    
    return fig

def create_population_vs_aqi(df, year):
    """Creates a scatter plot of population vs AQI metrics"""
    year_data = df[df['Year'] == year]
    
    fig = px.scatter(
        year_data,
        x='Pop_Est',
        y='Median AQI',
        size='Dys_w_AQI',
        color='Total_Unhealthy_Days',
        hover_name='State',
        log_x=True,
        size_max=30,
        title=f'Population vs. AQI ({year})',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title='Population (log scale)',
        yaxis_title='Median AQI',
        coloraxis_colorbar_title='Unhealthy Days'
    )
    
    return fig

def create_aqi_compliance_dashboard(df, year):
    """Creates a dashboard showing compliance with AQI standards"""
    year_data = df[df['Year'] == year]
    
    # Calculate compliance metrics
    year_data['Compliance_Rate'] = year_data['Good Days'] / year_data['Dys_w_AQI'] * 100
    
    # Create subplot layout
    fig = make_subplots(
        rows=2, 
        cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "bar", "colspan": 2}, None]],
        subplot_titles=("National Average Compliance Rate (%)", 
                        "States Meeting Standards (%)",
                        "Top 10 States by Compliance Rate")
    )
    
    # Add indicators
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=year_data['Compliance_Rate'].mean(),
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "green"},
                   'steps': [
                       {'range': [0, 50], 'color': "red"},
                       {'range': [50, 75], 'color': "yellow"},
                       {'range': [75, 100], 'color': "green"}
                   ]},
            title={'text': "National Average"}
        ),
        row=1, col=1
    )
    
    # Calculate percentage of states meeting a compliance threshold (e.g., 75%)
    threshold = 75
    pct_states_meeting = (year_data['Compliance_Rate'] >= threshold).mean() * 100
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=pct_states_meeting,
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "blue"},
                   'steps': [
                       {'range': [0, 33], 'color': "red"},
                       {'range': [33, 66], 'color': "yellow"},
                       {'range': [66, 100], 'color': "green"}
                   ]},
            title={'text': f"States ≥ {threshold}% Compliance"}
        ),
        row=1, col=2
    )
    
    # Add bar chart for top 10 states
    top_states = year_data.sort_values('Compliance_Rate', ascending=False).head(10)
    
    fig.add_trace(
        go.Bar(
            x=top_states['State'],
            y=top_states['Compliance_Rate'],
            marker_color='green'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=700,
        title_text=f"AQI Compliance Dashboard ({year})"
    )
    
    return fig

def create_unhealthy_days_choropleth(df, year):
    """Creates a choropleth map of unhealthy days by state"""
    year_data = df[df['Year'] == year]
    
    fig = px.choropleth(
        year_data,
        locations='State',
        locationmode="USA-states",
        color='Total_Unhealthy_Days',
        scope="usa",
        color_continuous_scale="Reds",
        hover_name='State',
        title=f'Total Unhealthy Air Days by State ({year})'
    )
    
    fig.update_layout(
        coloraxis_colorbar_title='Unhealthy Days'
    )
    
    return fig

def create_dominant_pollutant_map(df, year):
    """Creates a map showing the dominant pollutant by state"""
    year_data = df[df['Year'] == year]
    
    fig = px.choropleth(
        year_data,
        locations='State',
        locationmode="USA-states",
        color='Dominant_Pollutant',
        scope="usa",
        category_orders={'Dominant_Pollutant': ['CO', 'NO2', 'Ozone', 'PM2.5', 'PM10']},
        color_discrete_map={
            'CO': '#1f77b4',
            'NO2': '#ff7f0e',
            'Ozone': '#2ca02c',
            'PM2.5': '#d62728',
            'PM10': '#9467bd'
        },
        hover_name='State',
        hover_data=['Dominant_Pollutant', 'Days CO', 'Days NO2', 'Days Ozone', 'Days PM2.5', 'Days PM10'],
        title=f'Dominant Pollutant by State ({year})'
    )
    
    return fig

def create_aqi_health_impact_dashboard(df, year):
    """Creates visualizations related to health impacts of poor air quality"""
    year_data = df[df['Year'] == year]
    
    # Create a risk index (this would ideally be based on real health data)
    year_data['Risk_Index'] = (
        year_data['Unhealthy for Sensitive Groups Days'] * 1 +
        year_data['Unhealthy Days'] * 2 +
        year_data['Very Unhealthy Days'] * 3 +
        year_data['Hazardous Days'] * 5
    ) / year_data['Dys_w_AQI'] * 10
    
    # Create risk level categories
    bins = [0, 0.5, 1, 2, float('inf')]
    labels = ['Low', 'Moderate', 'High', 'Severe']
    year_data['Risk_Level'] = pd.cut(year_data['Risk_Index'], bins=bins, labels=labels)
    
    # Count states in each risk category
    risk_counts = year_data['Risk_Level'].value_counts().reset_index()
    risk_counts.columns = ['Risk_Level', 'Count']
    risk_counts = risk_counts.sort_values(by='Risk_Level', key=lambda x: pd.Categorical(
        x, categories=['Low', 'Moderate', 'High', 'Severe'], ordered=True
    ))
    
    # Create a pie chart for risk level distribution
    fig1 = px.pie(
        risk_counts, 
        names='Risk_Level', 
        values='Count',
        title=f'Distribution of AQI Health Risk Levels ({year})',
        color='Risk_Level',
        color_discrete_map={
            'Low': 'green',
            'Moderate': 'yellow',
            'High': 'orange',
            'Severe': 'red'
        }
    )
    
    # Create a map of risk levels
    fig2 = px.choropleth(
        year_data,
        locations='State',
        locationmode="USA-states",
        color='Risk_Level',
        scope="usa",
        category_orders={'Risk_Level': ['Low', 'Moderate', 'High', 'Severe']},
        color_discrete_map={
            'Low': 'green',
            'Moderate': 'yellow',
            'High': 'orange',
            'Severe': 'red'
        },
        hover_name='State',
        hover_data=['Risk_Index', 'Risk_Level', 'Unhealthy Days', 'Very Unhealthy Days', 'Hazardous Days'],
        title=f'AQI Health Risk Level by State ({year})'
    )
    
    return fig1, fig2