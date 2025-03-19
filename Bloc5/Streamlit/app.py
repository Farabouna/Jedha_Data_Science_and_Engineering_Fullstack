import streamlit as st
import pandas as pd
import openpyxl
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np


### CONFIG
st.set_page_config(
    page_title="Getaround Delay Analysis",
    page_icon="🚗",
    layout="wide"
  )

### TITLE AND TEXT
st.title("Getaround Delay Analysis")

st.markdown("The aim of this dashboard is to help the product manager to make decisions concerning:")
st.markdown("-How long should the minimum delay be between two rentals? (threshold?)")
st.markdown("-Should we enable the feature for all cars?, only Connect cars? (the scope?)")

### EXPANDER

with st.expander("⏯️ What is Getaround Connect?"):
    st.video("https://www.youtube.com/watch?v=3LyzwpGSfzE")

st.markdown("---")

### LOAD DATA
DATA_URL = ('get_around_delay_analysis.xlsx')
@st.cache_data # Caches the function output to avoid reloading data every time
def load_data():
    data = pd.read_excel(DATA_URL)
    return data

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("") # change text from "Loading data..." to "" once the the load_data function has run

### Run the below code if the check is checked ✅
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.markdown("---")

#### CREATE THREE COLUMNS
col1, col2, col3 = st.columns(3)

with col1:

    checkin_type_counts = data['checkin_type'].value_counts()
    fig = px.pie( 
    names=checkin_type_counts.index, 
    values=checkin_type_counts.values, 
    title='Check-in Type Percentages for Rentals',
    )
    
    fig.show()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("20% of the rentals use the Connect check-in type, the other 80% use the Mobile check-in.")

with col2:

    def get_checkin_type(car_id, data):
        """
        Returns the checkin type for a given car_id.

        Args:
            car_id (int): The ID of the car.
            data (pd.DataFrame): The DataFrame containing car data.

        Returns:
            str: The checkin type ('mobile', 'connect', or 'both').
        """

        checkin_types = data[data['car_id'] == car_id]['checkin_type'].unique()

        if len(checkin_types) == 1:
            return checkin_types[0]  # Return the unique checkin type
        elif len(checkin_types) > 1:
            return 'both'  # Return 'both' if multiple checkin types
        else:
            return None  # Return None if no checkin type found for the car_id

    # Apply the function to get checkin type for all cars
    data['checkin_type_category'] = data['car_id'].apply(lambda x: get_checkin_type(x, data))

    # Group by car_id and count the occurrences of checkin types
    checkin_type_counts = data.groupby('car_id')['checkin_type_category'].first().value_counts()

    fig = px.pie(
        names=checkin_type_counts.index,
        values=checkin_type_counts.values,
        title='Check-in Type Percentages for Cars'
    )

    fig.show()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("91% of the cars don't have the Connect check-in type.")

with col3:

    # Filter for cars with both check-in types
    cars_with_both_types = data[data.groupby('car_id')['checkin_type'].transform('nunique') > 1]['car_id'].unique()
    filtered_data = data[data['car_id'].isin(cars_with_both_types)]

    # Calculate check-in type percentages for filtered data
    checkin_type_percentages = filtered_data['checkin_type'].value_counts(normalize=True) * 100

    # Create a pie chart using Plotly Express
    fig = px.pie(
        names=checkin_type_percentages.index,
        values=checkin_type_percentages.values,
        title='Check-in Type for Cars with Both Types'
    )

    fig.show()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("When a car has both check-in types, 73% of the clients choose Connect check-in.")

st.markdown("📌 The Connect check-in type is clearly the preferred choice for rentals.")  

data = data.drop(index=21002) # Drop row with canceled state but with delay

st.markdown("---")

#### CREATE TWO COLUMNS
col1, col2 = st.columns(2)

with col1:

    # Filter out rows with missing 'delay_at_checkout_in_minutes'
    valid_delays_df = data[data['delay_at_checkout_in_minutes'].notna()]
    # Create the new column 'delay_at_checkout_type'
    valid_delays_df['delay_at_checkout_type'] = np.where(valid_delays_df['delay_at_checkout_in_minutes'] > 0, 'late',
                                        np.where(valid_delays_df['delay_at_checkout_in_minutes'] < 0, 'in advance',
                                        np.where(valid_delays_df['delay_at_checkout_in_minutes'] == 0 , 'no_delay',
                                        # Use np.nan instead of 'Nan' and provide a value for False condition
                                        np.where(valid_delays_df['delay_at_checkout_in_minutes'].isnull(), np.nan, np.nan))))
    # Count occurrences of type of checkout
    checkout_type_counts = valid_delays_df['delay_at_checkout_type'].value_counts()
    # Create a pie chart using Plotly Express
    fig = px.pie( 
    names=checkout_type_counts.index, 
    values=checkout_type_counts.values, 
    title='Delay Distribution'
    )
    
    fig.show()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("57% of the rentals are late and only 43% are on time or returned early.")

with col2:

    # Filter for late checkouts (positive delays)
    late_checkouts_df = data[data['delay_at_checkout_in_minutes'] > 0]

    # Define delay intervals
    delay_intervals = [
        (1, 60),      # 1 to 60 minutes (1 hour)
        (61, 120),    # 61 to 120 minutes (2 hours)
        (121, 180),   # 121 to 180 minutes (3 hours)
        (181, 240),   # 181 to 240 minutes (4 hours)
        (241, 300),   # 141 to 300 minutes (5 hours)
        (301, float('inf'))  # More than 300 minutes
    ]

    # Create a list to store the percentages
    delay_percentages = []

    # Calculate percentage for each interval
    for interval in delay_intervals:
        lower_bound, upper_bound = interval
        # Count delays within the interval
        delay_count = late_checkouts_df[
            (late_checkouts_df['delay_at_checkout_in_minutes'] >= lower_bound) &
            (late_checkouts_df['delay_at_checkout_in_minutes'] <= upper_bound)
        ].shape[0]
        # Calculate percentage
        percentage = (delay_count / late_checkouts_df.shape[0]) * 100
        delay_percentages.append(percentage)

    # Create a DataFrame to display the results
    delay_distribution_only_df = pd.DataFrame({
        'Delay Interval (minutes)': [f"{lower}-{upper}" for lower, upper in delay_intervals],
        'Percentage of Delays': delay_percentages
    })

    # Create a pie chart using Plotly Express
    fig = px.pie(
        delay_distribution_only_df,
        names='Delay Interval (minutes)',
        values='Percentage of Delays',
        title='Delay Distribution (Late Checkouts Only)'
    )
    
    fig.show()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("53% of late checkouts are under 1 hour, while 11% exceed 5 hours.")

st.markdown("📌 The first hour between two rentals is the most common delay period.")  

st.markdown("---")

#### CREATE TWO COLUMNS
col1, col2 = st.columns(2)

with col1:

    # Filter out rows with missing time_delta_with_previous_rental_in_minutes
    valid_time_delta_df = data[data['time_delta_with_previous_rental_in_minutes'].notna()]

    # Define time delta intervals
    time_delta_intervals = [
        (0, 0),       # 0 minutes
        (1, 60),      # 1 to 60 minutes (1 hour)
        (61, 120),    # 61 to 120 minutes (2 hours)
        (121, 180),   # 121 to 180 minutes (3 hours)
        (181, 240),   # 181 to 240 minutes (4 hours)
        (241, 300),   # 141 to 300 minutes (5 hours)
        (301, float('inf'))  # More than 300 minutes 
    ]

    # Create a list to store the percentages
    time_delta_percentages = []

    # Calculate percentage for each interval
    for interval in time_delta_intervals:
        lower_bound, upper_bound = interval
        # Count time deltas within the interval
        time_delta_count = valid_time_delta_df[
            (valid_time_delta_df['time_delta_with_previous_rental_in_minutes'] >= lower_bound) &
            (valid_time_delta_df['time_delta_with_previous_rental_in_minutes'] <= upper_bound)
        ].shape[0]
        # Calculate percentage using valid_time_delta_df.shape[0]
        percentage = (time_delta_count / valid_time_delta_df.shape[0]) * 100
        time_delta_percentages.append(percentage)

    # Create a DataFrame to display the results
    time_delta_distribution_df = pd.DataFrame({
        'Time Delta Interval (minutes)': [f"{lower}-{upper}" for lower, upper in time_delta_intervals],
        'Percentage of Rentals': time_delta_percentages
    })

    # Create a bar chart
    fig = px.bar(
        time_delta_distribution_df, 
        x='Time Delta Interval (minutes)', 
        y='Percentage of Rentals',
        title='Time Delta Distribution',
        text='Percentage of Rentals'  # Add text to display percentage values
    )

    # Update layout to display text on top of bars
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside') 
    fig.show()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("15% of rentals have no buffer and 17% have less than 1 hour delta.")

with col2:

    # Filter out rows with missing 'delay_at_checkout_in_minutes'
    valid_delays_df = data[data['delay_at_checkout_in_minutes'].notna()]
    # Define delay intervals
    delay_intervals = [
        (1, 60),      # 1 to 60 minutes (1 hour)
        (61, 120),    # 61 to 120 minutes (2 hours)
        (121, 180),   # 121 to 180 minutes (3 hours)
        (181, 240),   # 181 to 240 minutes (4 hours)
        (241, 300),   # 141 to 300 minutes (5 hours)
        (301, float('inf'))  # More than 300 minutes 
    ]

    # Create a list to store the percentages
    delay_percentages = []

    # Calculate percentage for each interval
    for interval in delay_intervals:
        lower_bound, upper_bound = interval
        # Count delays within the interval
        delay_count = valid_delays_df[
            (valid_delays_df['delay_at_checkout_in_minutes'] >= lower_bound) &
            (valid_delays_df['delay_at_checkout_in_minutes'] <= upper_bound)
        ].shape[0]
        # Calculate percentage
        percentage = (delay_count / valid_delays_df.shape[0]) * 100
        delay_percentages.append(percentage)

    # Create a DataFrame to display the results
    delay_distribution_df = pd.DataFrame({
        'Delay Interval (minutes)': [f"{lower}-{upper}" for lower, upper in delay_intervals],
        'Percentage of Delays': delay_percentages
    })

    # Create a bar chart
    fig = px.bar(
        delay_distribution_df, 
        x='Delay Interval (minutes)', 
        y='Percentage of Delays',
        title='Delay Distribution',
        text='Percentage of Delays'  # Add text to display percentage values
    )

    # Update layout to display text on top of bars
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside') 
    fig.show()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("31% of late checkouts are under an hour, while 6% exceed 5 hours.")

st.markdown("📌 This confirms that the first hour between two rentals is the most critical.")

st.markdown("---")

# Merge the DataFrame with itself to link rentals with their previous rentals
merged_data_checkin = pd.merge(data, data, left_on='rental_id', right_on='previous_ended_rental_id', how='inner', suffixes=('_current', '_previous'))

# Filter the merged data to keep only rows where rental_id matches previous_ended_rental_id
# Use the renamed column 'rental_id_current' instead of 'rental_id'
filtered_data = merged_data_checkin[merged_data_checkin['rental_id_current'] == merged_data_checkin['previous_ended_rental_id_previous']]

# Calculate checkin_delay_in_minutes for the filtered data
filtered_data['checkin_delay_in_minutes'] = filtered_data['time_delta_with_previous_rental_in_minutes_previous'] - filtered_data['delay_at_checkout_in_minutes_current']

#### CREATE TWO COLUMNS
col1, col2 = st.columns(2)

with col1:

    # Create categories based on 'checkin_delay_in_minutes' values (excluding NaN)
    filtered_data['checkin_delay_type'] = pd.cut(
    filtered_data['checkin_delay_in_minutes'],
    bins=[-float('inf'), 0, float('inf')],
    labels=['Negative time left', 'Positive time left'],
    include_lowest=True,
    right=False
    )

    # Filter out rows with NaN values in 'checkin_delay_type' (created from NaN in 'checkin_delay_in_minutes')
    filtered_data_no_nan = filtered_data[filtered_data['checkin_delay_type'].notna()]

    # Count occurrences of each category (excluding NaN)
    checkin_delay_counts = filtered_data_no_nan['checkin_delay_type'].value_counts()

    # Create a pie chart using Plotly Express
    fig = px.pie(
        names=checkin_delay_counts.index,
        values=checkin_delay_counts.values,
        title='Distribution of Check-in Delay Types',
    )

    fig.show()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("87% of check-ins have remaining time left, while 13% are delayed.")

with col2:
    
    # Filter for negative time left
    negative_time_left_df = filtered_data[filtered_data['checkin_delay_type'] == 'Negative time left']

    # Count occurrences of checkin_type within negative time left
    # Access the original 'checkin_type' column from the 'filtered_data' DataFrame
    checkin_type_counts = filtered_data.loc[negative_time_left_df.index, 'checkin_type_previous'].value_counts()

    # Create the pie chart
    fig = px.pie(
    names=checkin_type_counts.index,  # Check-in types (e.g., 'mobile', 'connect')
    values=checkin_type_counts.values,  # Counts for each check-in type
    title='Check-in Type Distribution for Negative Time Left',
    )

    fig.show()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("68% of the delayed check-ins use the Mobile check-in type.")

st.markdown("📌 The Mobile check-in type appears to be the most common for delayed check-ins.")  

st.markdown("---")

#### CREATE TWO COLUMNS
col1, col2 = st.columns(2)

with col1:

    # Filter for negative time left
    negative_time_left_df = filtered_data[filtered_data['checkin_delay_type'] == 'Negative time left']
    # Group by 'state_previous' and 'checkin_type_previous' and get counts
    grouped_counts = negative_time_left_df.groupby(['state_previous', 'checkin_type_previous']).size().reset_index(name='count')
    # Create pie chart for 'mobile' check-in type
    mobile_data = grouped_counts[grouped_counts['checkin_type_previous'] == 'mobile']
    fig = px.pie(
    mobile_data,
    names='state_previous',
    values='count',
    title='Distribution of Rental State for Mobile Check-ins',
    )
    fig.show()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("88% of the delayed Mobile check-in ended while 12% were cancelled.")

with col2:

    # Create pie chart for 'connect' check-in type
    connect_data = grouped_counts[grouped_counts['checkin_type_previous'] == 'connect']
    fig = px.pie(
    connect_data,
    names='state_previous',
    values='count',
    title='Distribution of Rental State for Connect Check-ins',
    )
    fig.show()
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("72% of the delayed Connect check-ins ended while 28% were cancelled.")

st.markdown("📌 While Mobile Check-in is the most common type for delayed check-ins, Connect Check-in appears to have a higher cancellation rate.")  

st.markdown("---")

st.markdown("### 📌 Take-Home Message")

st.markdown("Increasing the **minimum delta time** between rentals significantly reduces delays, but at the cost of losing some rentals. Below is a summary of the trade-offs:")

col1, col2 = st.columns(2)  # Create two columns

with col1:
    st.markdown("#### ⏳ 1 Hour")
    st.markdown("- 🚗 **Delays reduced**: **53.4%**")
    st.markdown("- ⚠️ **Remaining delays**: **26.8%**")
    st.markdown("- ❌ **Rentals lost**: **31.8%**")

    st.markdown("#### ⏳ 3 Hours")
    st.markdown("- 🚗 **Delays reduced**: **81.4%**")
    st.markdown("- ⚠️ **Remaining delays**: **10.7%**")
    st.markdown("- ❌ **Rentals lost**: **51.8%**")

    st.markdown("#### ⏳ 5 Hours")
    st.markdown("- 🚗 **Delays reduced**: **88.9%**")
    st.markdown("- ⚠️ **Remaining delays**: **6.4%**")
    st.markdown("- ❌ **Rentals lost**: **62.1%**")

with col2:
    st.markdown("#### ⏳ 2 Hours")
    st.markdown("- 🚗 **Delays reduced**: **72.9%**")
    st.markdown("- ⚠️ **Remaining delays**: **15.6%**")
    st.markdown("- ❌ **Rentals lost**: **43.7%**")

    st.markdown("#### ⏳ 4 Hours")
    st.markdown("- 🚗 **Delays reduced**: **85.9%**")
    st.markdown("- ⚠️ **Remaining delays**: **8.1%**")
    st.markdown("- ❌ **Rentals lost**: **58%**")

st.markdown("Although delays have been reduced, the loss of rentals remains significant. The optimal minimum delta time appears to be 1 hour.")  

st.markdown("Regarding check-in types, Mobile Check-in is the most common and also the most delayed. This method requires additional time between rentals to complete the check-in and check-out process, sign the rental agreement, and necessitates the owner's presence. Therefore, Mobile Check-in demands extra attention and may benefit from a longer minimum delta time.")  