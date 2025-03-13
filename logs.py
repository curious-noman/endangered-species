import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Title and description
st.title("Endangered Animal Detection Logs")
st.write("View and analyze the history of animal detections")

# Initialize logs from session state if it doesn't exist
if 'detection_logs' not in st.session_state:
    st.session_state.detection_logs = []

# Get logs from session state
logs = st.session_state.detection_logs

if not logs:
    st.warning("No detections recorded yet. Start detecting animals first.")
else:
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(logs)

    # Convert timestamp strings to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp (newest first)
    df = df.sort_values('timestamp', ascending=False)

    # Add a date column for filtering
    df['date'] = df['timestamp'].dt.date

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(
            df['date'].min(),
            df['date'].max()
        ),
        min_value=df['date'].min(),
        max_value=df['date'].max()
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        filtered_df = df.loc[mask]
    else:
        filtered_df = df

    # Category filter
    categories = ['All'] + sorted(df['category'].unique().tolist())
    selected_category = st.sidebar.selectbox("Select Category", categories)

    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]

    # Confidence score filter
    min_confidence = st.sidebar.slider(
        "Minimum Confidence Score (%)",
        min_value=90,
        max_value=100,
        value=90,
        step=1
    )

    filtered_df = filtered_df[filtered_df['confidence_score'] * 100 >= min_confidence]

    # Display summary metrics
    st.header("Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Detections", len(filtered_df))

    with col2:
        if not filtered_df.empty:
            st.metric("Unique Species", filtered_df['class_name'].nunique())
        else:
            st.metric("Unique Species", 0)

    with col3:
        if not filtered_df.empty:
            avg_confidence = (filtered_df['confidence_score'] * 100).mean()
            st.metric("Avg. Confidence Score", f"{avg_confidence:.2f}%")
        else:
            st.metric("Avg. Confidence Score", "N/A")

    # Visualizations
    if not filtered_df.empty:
        st.header("Visualizations")

        tab1, tab2 = st.tabs(["Detection Trends", "Category Distribution"])

        with tab1:
            # Group by date and count detections
            daily_counts = filtered_df.groupby(filtered_df['timestamp'].dt.date).size().reset_index(name='count')
            daily_counts.columns = ['date', 'count']

            # Create the time trend chart
            fig = px.line(
                daily_counts,
                x='date',
                y='count',
                title='Daily Detection Count',
                labels={'date': 'Date', 'count': 'Number of Detections'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Create a pie chart of categories
            category_counts = filtered_df['category'].value_counts().reset_index()
            category_counts.columns = ['category', 'count']

            fig = px.pie(
                category_counts,
                values='count',
                names='category',
                title='Detection Categories'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Display the data table
    st.header("Detection Log")

    # Format the dataframe for display
    display_df = filtered_df.copy()
    display_df['confidence_score'] = (display_df['confidence_score'] * 100).round(2).astype(str) + '%'
    display_df = display_df[['timestamp', 'class_name', 'category', 'confidence_score']]
    display_df.columns = ['Timestamp', 'Species', 'Category', 'Confidence Score']

    st.dataframe(display_df, use_container_width=True)

    # Download option
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download Detection Log CSV",
        data=csv,
        file_name="animal_detection_log.csv",
        mime="text/csv",
    )

    # Add a note about session state
    st.caption("Note: Detection logs are stored in session state and will be reset if you refresh the page or if the Streamlit server restarts.")