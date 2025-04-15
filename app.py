import streamlit as st
import pandas as pd
from utils.data_loader import load_data
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
from utils.visualization import (
    create_yearly_aqi_heatmap, create_aqi_distribution, create_pollutant_radar_chart,
    create_aqi_forecast, create_seasonal_analysis, create_population_vs_aqi,
    create_aqi_compliance_dashboard, create_unhealthy_days_choropleth,
    create_dominant_pollutant_map, create_aqi_health_impact_dashboard
)

st.set_page_config(
    page_title="AQI Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def get_data():
    return load_data()

def main():
    df = get_data()
    st.sidebar.title("AQI Dashboard")
    st.sidebar.image("pic.png")
    
    pages = {
        "Home": "Overview and summary statistics",
        "Trends Analysis": "Historical AQI trends over time",
        "State Comparison": "Compare AQI metrics between states",
        "Map View": "Geospatial visualization of AQI data",
        "Pollutant Analysis": "Analysis of specific air pollutants",
        "Health Impact": "Health impact analysis based on AQI data",
        "Forecasting": "Future AQI trend predictions",
        "Compliance Dashboard": "AQI compliance analysis across states"
    }
    
    st.sidebar.link_button("Dataset Link", url = "https://www.kaggle.com/datasets/adampq/air-quality-index-by-state-1980-2022/")
    st.sidebar.header("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    st.sidebar.info(pages[selection])
    
    if selection == "Home":
        home_page(df)
    elif selection == "Trends Analysis":
        trends_page(df)
    elif selection == "State Comparison":
        state_comparison_page(df)
    elif selection == "Map View":
        map_view_page(df)
    elif selection == "Pollutant Analysis":
        pollutant_page(df)
    elif selection == "Health Impact":
        health_impact_page(df)
    elif selection == "Forecasting":
        forecasting_page(df)
    elif selection == "Compliance Dashboard":
        compliance_page(df)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("AQI Dashboard © 2025")

def home_page(df):
    st.title("Air Quality Index Dashboard")
    st.subheader("Overview of AQI Data (1980-2022)")
    st.write("This dashboard provides comprehensive insights into Air Quality Index (AQI) data across the United States from 1980 to 2022.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Years of Data", f"{df['Year'].nunique()}")
    with col2:
        st.metric("States Covered", f"{df['State'].nunique()}")
    with col3:
        st.metric("Total Records", f"{len(df):,}")
    with col4:
        st.metric("Avg. Reporting Rate", f"{df['Reporting_Rate'].mean():.1f}%")
    
    # New visualization - National AQI trends
    st.subheader("National AQI Trends Over Time")
    yearly_avg = df.groupby('Year').agg({
        'Median AQI': 'mean',
        'Max AQI': 'mean',
        '90th Percentile AQI': 'mean'
    }).reset_index()
    fig_trend = px.line(
        yearly_avg,
        x="Year",
        y=['Median AQI', 'Max AQI', '90th Percentile AQI'],
        title="National Average AQI Trends (1980-2022)",
        markers=True
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Most recent year summary
    st.subheader("Most Recent Year Summary")
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year]
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_aqi_states = latest_data.sort_values('Median AQI', ascending=False).head(10)
        fig = px.bar(
            top_aqi_states,
            x='State',
            y='Median AQI',
            color='Median AQI',
            title=f"Top 10 States by Median AQI ({latest_year})",
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Add a new visualization for the home page
        top_unhealthy_states = latest_data.sort_values('Total_Unhealthy_Days', ascending=False).head(10)
        fig2 = px.bar(
            top_unhealthy_states,
            x='State',
            y='Total_Unhealthy_Days',
            color='Total_Unhealthy_Days',
            title=f"Top 10 States by Unhealthy Days ({latest_year})",
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Dominant pollutant distribution
    st.subheader(f"Dominant Pollutant Distribution ({latest_year})")
    pollutant_counts = latest_data['Dominant_Pollutant'].value_counts().reset_index()
    pollutant_counts.columns = ['Pollutant', 'Count']
    
    fig3 = px.pie(
        pollutant_counts,
        names='Pollutant',
        values='Count',
        title=f"Distribution of Dominant Pollutants ({latest_year})",
        color='Pollutant',
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Sample data
    with st.expander("View Sample Data"):
        st.dataframe(df.head())

def trends_page(df):
    st.title("AQI Trends Over Time")
    
    selected_states = st.multiselect(
        "Select States",
        options=sorted(df['State'].unique()),
        default=["California", "New York", "Texas"]
    )
    
    metric = st.selectbox(
        "Select Metric",
        options=[
            "Median AQI", 
            "Max AQI", 
            "90th Percentile AQI",
            "Good Days",
            "Unhealthy Days",
            "Total_Unhealthy_Days"
        ],
        index=0
    )
    
    show_national_avg = st.checkbox("Show National Average", value=True)
    
    if not selected_states:
        st.warning("Please select at least one state to visualize")
    else:
        filtered_df = df[df['State'].isin(selected_states)]
        
        # Create the line chart for selected states
        fig = px.line(
            filtered_df,
            x="Year",
            y=metric,
            color="State",
            title=f"{metric} Trends by State (1980-2022)",
            markers=True,
            line_shape="linear"
        )
        
        # Add national average if requested
        if show_national_avg:
            national_avg = df.groupby('Year')[metric].mean().reset_index()
            fig.add_trace(
                go.Scatter(
                    x=national_avg['Year'],
                    y=national_avg[metric],
                    mode='lines',
                    name='National Average',
                    line=dict(color='black', width=2, dash='dash')
                )
            )
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title=metric,
            legend_title="State"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Add heatmap visualization
    st.subheader("AQI Metrics Heatmap")
    heatmap_fig = create_yearly_aqi_heatmap(df)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    st.subheader("National Average Trends")
    
    yearly_avg = df.groupby('Year').agg({
        'Median AQI': 'mean',
        'Max AQI': 'mean',
        '90th Percentile AQI': 'mean',
        'Good Days': 'mean',
        'Moderate Days': 'mean',
        'Unhealthy Days': 'mean',
        'Very Unhealthy Days': 'mean',
        'Hazardous Days': 'mean',
        'Total_Unhealthy_Days': 'mean'
    }).reset_index()
    
    metrics = st.multiselect(
        "Select Metrics to Compare",
        options=[
            'Median AQI', 
            'Max AQI', 
            '90th Percentile AQI',
            'Good Days', 
            'Moderate Days', 
            'Unhealthy Days',
            'Very Unhealthy Days',
            'Hazardous Days',
            'Total_Unhealthy_Days'
        ],
        default=['Median AQI']
    )
    
    if metrics:
        fig2 = px.line(
            yearly_avg,
            x="Year",
            y=metrics,
            title="National Average Trends Over Time",
            markers=True
        )
        fig2.update_layout(
            xaxis_title="Year",
            yaxis_title="Value",
            legend_title="Metric"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Add new section for distribution analysis
    st.subheader("AQI Distribution Analysis")
    dist_year = st.slider(
        "Select Year for Distribution Analysis",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=int(df['Year'].max()),
        step=1
    )
    
    dist_fig = create_aqi_distribution(df, dist_year)
    st.plotly_chart(dist_fig, use_container_width=True)

def state_comparison_page(df):
    st.title("State-by-State AQI Comparison")
    
    selected_year = st.slider(
        "Select Year",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=int(df['Year'].max()),
        step=1
    )
    
    year_data = df[df['Year'] == selected_year]
    
    # Add tabs for different comparison views
    tab1, tab2, tab3 = st.tabs(["Metric Comparison", "Correlation Analysis", "Pollutant Radar"])
    
    with tab1:
        metric = st.selectbox(
            "Select Metric to Compare",
            options=[
                "Median AQI",
                "Max AQI", 
                "90th Percentile AQI",
                "Good Days",
                "Moderate Days",
                "Unhealthy for Sensitive Groups Days",
                "Unhealthy Days",
                "Very Unhealthy Days",
                "Hazardous Days",
                "Total_Unhealthy_Days",
                "Reporting_Rate"
            ],
            index=0
        )
        
        fig = px.bar(
            year_data.sort_values(metric, ascending=False),
            x='State',
            y=metric,
            color=metric,
            title=f"{metric} by State in {selected_year}",
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        fig.update_layout(
            xaxis_title="State",
            yaxis_title=metric,
            xaxis={'categoryorder':'total descending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Correlation Between Metrics")
        
        corr_cols = [
            'Median AQI', 'Max AQI', '90th Percentile AQI',
            'Good Days', 'Moderate Days', 'Unhealthy for Sensitive Groups Days',
            'Unhealthy Days', 'Very Unhealthy Days', 'Hazardous Days',
            'Total_Unhealthy_Days', 'Reporting_Rate'
        ]
        
        corr_matrix = year_data[corr_cols].corr()
        
        fig2 = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmin=-1, zmax=1
        ))
        
        fig2.update_layout(
            title=f"Correlation Between AQI Metrics ({selected_year})",
            height=600
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("Pollutant Comparison")
        radar_states = st.multiselect(
            "Select States to Compare",
            options=sorted(year_data['State'].unique()),
            default=["California", "New York", "Texas"],
            max_selections=5
        )
        
        if radar_states:
            radar_fig = create_pollutant_radar_chart(df, radar_states, selected_year)
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Please select states to compare pollutant profiles")
    
    # Add population vs AQI relationship
    st.subheader("Population vs AQI Relationship")
    pop_fig = create_population_vs_aqi(df, selected_year)
    st.plotly_chart(pop_fig, use_container_width=True)

def map_view_page(df):
    st.title("Geospatial AQI Visualization")
    
    selected_year = st.slider(
        "Select Year",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=int(df['Year'].max()),
        step=1
    )
    
    year_data = df[df['Year'] == selected_year]
    
    metric = st.selectbox(
        "Select Metric to Visualize",
        options=[
            "Median AQI",
            "Max AQI", 
            "90th Percentile AQI",
            "Good Days",
            "Unhealthy Days",
            "Very Unhealthy Days",
            "Total_Unhealthy_Days"
        ],
        index=0
    )
    
    def extract_coords(geo_str):
        pattern = r"POINT \(([-\d.]+) ([-\d.]+)\)"
        match = re.search(pattern, geo_str)
        if match:
            lon, lat = match.groups()
            return float(lon), float(lat)
        return None, None
    
    year_data['lon'], year_data['lat'] = zip(*year_data['Geo_Loc'].apply(extract_coords))
    
    fig = px.scatter_geo(
            year_data,
            lat='lat',
            lon='lon',
            scope="usa",
            color=metric,
            hover_name="State",
            size=metric,
            projection="albers usa",
            title=f"{metric} by State ({selected_year})",
            hover_data={
                "lat": False,
                "lon": False,
                metric: True,
                "Pop_Est": True,
                "Dys_w_AQI": True
            },
            color_continuous_scale=px.colors.sequential.Plasma
        )
    st.plotly_chart(fig, use_container_width=True)

def pollutant_page(df):
    st.title("Pollutant Analysis")
    
    selected_state = st.selectbox(
        "Select State",
        options=sorted(df['State'].unique()),
        index=0
    )
    
    year_range = st.slider(
        "Select Years Range",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=(int(df['Year'].min()), int(df['Year'].max())),
        step=1
    )

    filtered_df = df[(df['State'] == selected_state) & 
                        (df['Year'] >= year_range[0]) & 
                        (df['Year'] <= year_range[1])]
    
    # Summary metrics
    latest_year_data = filtered_df[filtered_df['Year'] == filtered_df['Year'].max()]
    if not latest_year_data.empty:
        latest_year = latest_year_data['Year'].iloc[0]
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        with metrics_col1:
            st.metric("Dominant Pollutant", latest_year_data['Dominant_Pollutant'].iloc[0])
        with metrics_col2:
            st.metric(f"Ozone Days ({latest_year})", f"{latest_year_data['Days Ozone'].iloc[0]:.0f}")
        with metrics_col3:
            st.metric(f"PM2.5 Days ({latest_year})", f"{latest_year_data['Days PM2.5'].iloc[0]:.0f}")
        with metrics_col4:
            st.metric(f"PM10 Days ({latest_year})", f"{latest_year_data['Days PM10'].iloc[0]:.0f}")
    
    # Line chart visualization
    pollutant_cols = ['Days CO', 'Days NO2', 'Days Ozone', 'Days PM2.5', 'Days PM10']
    pollutant_df = filtered_df.melt(
        id_vars=['Year'],
        value_vars=pollutant_cols,
        var_name='Pollutant',
        value_name='Days'
    )
    
    fig = px.line(
        pollutant_df,
        x="Year",
        y="Days",
        color="Pollutant",
        title=f"Days with Each Pollutant as Main Contributor in {selected_state} ({year_range[0]}-{year_range[1]})",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Days",
        legend_title="Pollutant"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Area chart for pollutant composition
    fig2 = px.area(
        pollutant_df,
        x="Year",
        y="Days",
        color="Pollutant",
        title=f"Composition of Pollutants in {selected_state} ({year_range[0]}-{year_range[1]})",
        line_shape="spline"
    )
    
    fig2.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Days",
        legend_title="Pollutant"
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Enhanced pollutant analysis - percentage view
    st.subheader("Pollutant Contribution by Percentage")
    yearly_pollutants = filtered_df.groupby('Year')[pollutant_cols].sum().reset_index()
    yearly_pollutants['Total'] = yearly_pollutants[pollutant_cols].sum(axis=1)
    
    for col in pollutant_cols:
        yearly_pollutants[f"{col} %"] = yearly_pollutants[col] / yearly_pollutants['Total'] * 100
    
    pct_cols = [f"{col} %" for col in pollutant_cols]
    pollutant_pct_df = yearly_pollutants.melt(
        id_vars=['Year'],
        value_vars=pct_cols,
        var_name='Pollutant',
        value_name='Percentage'
    )
    pollutant_pct_df['Pollutant'] = pollutant_pct_df['Pollutant'].str.replace(' %', '').str.replace('Days ', '')
    
    fig3 = px.area(
        pollutant_pct_df,
        x="Year",
        y="Percentage",
        color="Pollutant",
        title=f"Relative Contribution of Pollutants in {selected_state} ({year_range[0]}-{year_range[1]})",
        line_shape="spline"
    )
    
    fig3.update_layout(
        xaxis_title="Year",
        yaxis_title="Percentage (%)",
        legend_title="Pollutant",
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Correlation matrix between pollutants and AQI
    st.subheader(f"Relationship Between Pollutants and AQI in {selected_state}")
    
    pollutant_aqi_cols = pollutant_cols + ['Median AQI', 'Max AQI']
    
    with st.expander("View Correlation Matrix"):
        fig4 = px.scatter_matrix(
            filtered_df,
            dimensions=pollutant_aqi_cols,
            color="Year",
            title=f"Scatter Matrix of Pollutants vs AQI in {selected_state}"
        )
        
        fig4.update_layout(
            height=800
        )
        
        st.plotly_chart(fig4, use_container_width=True)

def health_impact_page(df):
    st.title("Health Impact Analysis")
    st.write("""
    This section provides analysis of potential health impacts based on AQI data.
    Air quality directly affects public health, with higher AQI values and more unhealthy days
    associated with increased respiratory and cardiovascular issues.
    """)
    
    selected_year = st.slider(
        "Select Year",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=int(df['Year'].max()),
        step=1
    )
    
    # Create health impact visualizations
    fig1, fig2 = create_aqi_health_impact_dashboard(df, selected_year)
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Add more health-related analysis
    st.subheader("State Health Impact Rankings")
    year_data = df[df['Year'] == selected_year]
    
    # Calculate health impact score (this is a proxy - in a real app would use health data)
    year_data['Health_Impact_Score'] = (
        year_data['Unhealthy for Sensitive Groups Days'] * 1 + 
        year_data['Unhealthy Days'] * 2 + 
        year_data['Very Unhealthy Days'] * 5 + 
        year_data['Hazardous Days'] * 10
    ) / year_data['Dys_w_AQI']
    
    # Show top states with highest health impact
    top_impact_states = year_data.sort_values('Health_Impact_Score', ascending=False).head(10)
    
    fig3 = px.bar(
        top_impact_states,
        x='State',
        y='Health_Impact_Score',
        color='Health_Impact_Score',
        title=f"Top 10 States by Estimated Health Impact ({selected_year})",
        color_continuous_scale='Reds'
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Add an analysis of sensitive populations
    st.subheader("Impact on Sensitive Populations")
    sensitive_days = year_data.sort_values('Unhealthy for Sensitive Groups Days', ascending=False).head(10)
    
    fig4 = px.bar(
        sensitive_days,
        x='State',
        y='Unhealthy for Sensitive Groups Days',
        color='Unhealthy for Sensitive Groups Days',
        title=f"States with Most Days Unhealthy for Sensitive Groups ({selected_year})",
        color_continuous_scale='Oranges'
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("""
    ## Interpreting Health Impact Scores
    
    The Health Impact Score is calculated based on the frequency and severity of unhealthy air days:
    - Each "Unhealthy for Sensitive Groups" day contributes 1 point
    - Each "Unhealthy" day contributes 2 points
    - Each "Very Unhealthy" day contributes 5 points
    - Each "Hazardous" day contributes 10 points
    
    The total is then normalized by the number of days with AQI measurements to account for 
    different reporting frequencies across states.
    
    Higher scores indicate greater potential health impacts from poor air quality.
    """)

def forecasting_page(df):
    st.title("AQI Forecasting")
    st.write("""
    This section provides simple forecasts of future AQI trends based on historical data.
    These projections use polynomial regression models and should be interpreted as general
    trend indicators rather than precise predictions.
    """)
    
    selected_state = st.selectbox(
        "Select State for Forecast",
        options=sorted(df['State'].unique()),
        index=0
    )
    
    forecast_metric = st.selectbox(
        "Select Metric to Forecast",
        options=[
            "Median AQI",
            "Max AQI",
            "Good Days",
            "Unhealthy Days",
            "Total_Unhealthy_Days"
        ],
        index=0
    )
    
    forecast_periods = st.slider(
        "Forecast Years Ahead",
        min_value=1,
        max_value=10,
        value=5
    )
    
    forecast_fig = create_aqi_forecast(
        df, 
        state=selected_state, 
        metric=forecast_metric, 
        periods=forecast_periods
    )
    
    st.plotly_chart(forecast_fig, use_container_width=True)

    # Add seasonal analysis
    st.subheader("Seasonal AQI Patterns")
    st.write("""
    The following visualization shows simulated seasonal patterns in AQI data.
    Note: This is a simulation based on typical seasonal variations, as the dataset
    doesn't contain month-level information.
    """)
    
    seasonal_fig = create_seasonal_analysis(df, selected_state)
    st.plotly_chart(seasonal_fig, use_container_width=True)
    
    # Add forecasting for multiple states
    st.subheader("State Comparison Forecasts")
    
    compare_states = st.multiselect(
        "Select States to Compare",
        options=sorted(df['State'].unique()),
        default=["California", "New York", "Texas"],
        max_selections=5
    )
    
    if compare_states:
        # Create subplot for multiple forecasts
        fig = go.Figure()
        
        for state in compare_states:
            state_data = df[df['State'] == state].sort_values('Year')
            
            # Simple polynomial fit
            X = state_data['Year'].values
            y = state_data[forecast_metric].values
            
            degree = 2
            coeffs = np.polyfit(X, y, degree)
            p = np.poly1d(coeffs)
            
            # Forecast years
            last_year = state_data['Year'].max()
            future_years = np.array(range(last_year + 1, last_year + forecast_periods + 1))
            forecast_values = p(future_years)
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=X,
                y=y,
                mode='markers+lines',
                name=f"{state} (Historical)",
                opacity=0.7
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=future_years,
                y=forecast_values,
                mode='lines',
                name=f"{state} (Forecast)",
                line=dict(dash='dash')
            ))
        
        fig.update_layout(
            title=f"Comparative {forecast_metric} Forecasts",
            xaxis_title="Year",
            yaxis_title=forecast_metric
        )
        
        st.plotly_chart(fig, use_container_width=True)

def compliance_page(df):
    st.title("AQI Compliance Dashboard")
    st.write("""
    This dashboard provides insights into how well states are meeting air quality standards
    based on the number of good air quality days and compliance with federal standards.
    """)
    
    selected_year = st.slider(
        "Select Year",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=int(df['Year'].max()),
        step=1
    )
    
    # Create compliance dashboard
    compliance_fig = create_aqi_compliance_dashboard(df, selected_year)
    st.plotly_chart(compliance_fig, use_container_width=True)
    
    # Add compliance trends over time
    st.subheader("Compliance Trends Over Time")
    
    # Calculate yearly compliance metrics
    yearly_compliance = df.groupby('Year').agg({
        'Good Days': 'mean',
        'Dys_w_AQI': 'mean'
    }).reset_index()
    
    yearly_compliance['Compliance_Rate'] = (yearly_compliance['Good Days'] / 
                                           yearly_compliance['Dys_w_AQI'] * 100)
    
    fig = px.line(
        yearly_compliance,
        x='Year',
        y='Compliance_Rate',
        title='National Average Compliance Rate Over Time',
        markers=True
    )
    
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Compliance Rate (%)',
        yaxis=dict(range=[0, 100])
    )
    
    # Add a reference line for the target compliance rate
    target_compliance = 75
    fig.add_hline(
        y=target_compliance,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Target: {target_compliance}%",
        annotation_position="bottom right"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # State compliance comparison
    st.subheader(f"State Compliance Comparison ({selected_year})")
    
    year_data = df[df['Year'] == selected_year].copy()
    year_data['Compliance_Rate'] = year_data['Good Days'] / year_data['Dys_w_AQI'] * 100
    
    # Sort states by compliance rate
    sorted_states = year_data.sort_values('Compliance_Rate', ascending=False)
    
    fig2 = px.bar(
        sorted_states,
        x='State',
        y='Compliance_Rate',
        color='Compliance_Rate',
        title=f'State Compliance Rates ({selected_year})',
        color_continuous_scale='RdYlGn',
        range_color=[0, 100]
    )
    
    # Add target line
    fig2.add_hline(
        y=target_compliance,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Target: {target_compliance}%",
        annotation_position="bottom right"
    )
    
    fig2.update_layout(
        xaxis_title='State',
        yaxis_title='Compliance Rate (%)',
        xaxis={'categoryorder':'total descending'},
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()