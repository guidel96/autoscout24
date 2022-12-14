# To run in a conda prompt with the right environment: 
# conda activate speech_analytics
# streamlit run streamlit.py


import pandas as pd
import numpy as np
import streamlit as st

#######################################################################################################################
####### STREAMLIT AUTOMATED DF SELECTION - SPECIFIC UI ################################################################
#######################################################################################################################

# Documentation: https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters", key='add_filters')

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

#######################################################################################################################
############# DATA PREPROCESSING ######################################################################################
#######################################################################################################################


df = pd.read_csv('./Used_cars.csv')

df = df.astype({
    'Brand': 'string',
    'Car_type': 'string',
    'Brandname_and_model': 'string', 
    'Model_type': 'string',
    'Price': 'int64', 
    'Price2': 'string', 
    'Seller_type': 'category', 
    'Selling_type': 'category', 
    'Listing_country': 'category',
    'Listing_zip_code': 'int64', 
    'Nb_previous_owners': 'string', 
    'Mileage': 'int64', 
    'Mileage2': 'string',
    'Power': 'category', 
    'Transmission': 'category', 
    'Fuel_type': 'category', 
    'Fuel_type2': 'category', 
    'Fuel_consumption': 'string',
    'Fuel_emissions': 'string', 
    'First_registration': 'string', 
    'First_registration2': 'string',
    'specific_model': 'category', 
    'Power_CH': 'int64'
}, errors='ignore')


#######################################################################################################################
############# STREAMLIT INTEGRATION ###################################################################################
#######################################################################################################################

st.set_page_config(page_title="Autoscout24", layout="wide")

#st.markdown("# Data Visualization")
#st.dataframe(df, width=None, height=None, use_container_width=False)

#st.markdown(" ## Auto-filter dataframe")
# st.dataframe(filter_dataframe(df), width = 1500, height=1000)


# function that asks the user to insert the column (names) he wants the dataframe to display
def select_columns(df):
    
    display_cols = ['Car_type','Model_type','specific_model','Price','Fuel_type2','Selling_type','Mileage','Power','Transmission','Fuel_emissions'] # starts with empty list of cols, or maybe start with a default selection ???

    modify = st.checkbox("Add Columns", key='add_cols')

    if not modify:          # If no new column is added by the user, we keep the same dataframe
        return display_cols # df

    df = df.copy()

    """Select the columns you would like to add to the dataframe"""
    # Inserts an invisible container into your app that can be used to hold multiple elements. 
    # This allows you to, for example, insert multiple elements into your app out of order.
    # To add elements to the returned container, you can use "with" notation (preferred) or just call methods directly on the returned object. 
    # See examples below.

    modification_container = st.container() 

    with modification_container:
        to_select_columns = st.multiselect(label="Select columns:", options=df.columns, default=display_cols)
        
        display_cols = to_select_columns
    
    
    return display_cols


# Display the DF with the cols selected by the user
#st.markdown(" ## Selected cols for dataframe")
#st.dataframe(df[select_columns(df)])
    

st.markdown(" ## DataFrame")
# start with df
cols = select_columns(df)
data = filter_dataframe(df[cols])           # this is the dataframe that is going to be displayed

# we add some pandas styling to the columns (e.g. 1.4000 becomes 1.4)
styled_data = data.style.format({
    'specific_model':'{:.1f}',       #displays only one decimal
    'Price': '{:,.0f}',                 # displays 10000 into 10.000
    'Mileage':'{:,.0f}'
    })                        
st.dataframe(styled_data, width = 1500, height=1000)


#######################################################################################################################
############# STREAMLIT VISUALS ###################################################################################
#######################################################################################################################

import plotly.express as px
import math

st.markdown('# Visualization')

fig_scatter_matrix = px.scatter_matrix(
    df,
    dimensions=['Price','Mileage','Power','specific_model'],
    width=1200, height=600
    )
#fig.show()
st.plotly_chart(fig_scatter_matrix, use_container_width=False, sharing='streamlit')

################################################################################################
# distribution of Mileage
bin_width = 20000 #km # cannot decide on a bin size, but can decide on the number of bins (and we can compute how many bins we need for a specific bin size)
nbins = math.ceil((df["Mileage"].max() - df["Mileage"].min()) / bin_width)
mileage_distr = px.histogram(df, x='Mileage', nbins=nbins, title='Histogram of Distance (per 20 km)')
st.plotly_chart(mileage_distr, sharing='streamlit')

# distribution of price
bin_width = 499 #km # cannot decide on a bin size, but can decide on the number of bins (and we can compute how many bins we need for a specific bin size)
nbins = math.ceil((df["Price"].max() - df["Price"].min()) / bin_width)
price_distr = px.histogram(df, x='Price', nbins=nbins, title='Distribution of Price')
st.plotly_chart(price_distr, sharing='streamlit')

# hist plot Power
power_hist = px.histogram(df, x='Power', title='Histogram of Power',category_orders=dict(Power=["51 kW (69 CH)","55 kW (75 CH)","66 kW (90 CH)"]))
st.plotly_chart(power_hist, sharing='streamlit')

# hist plot specific_model
specific_model_hist = px.histogram(df, x='specific_model', title='Histogram of Model') #,category_orders=dict(specific_model=["1.2","1.4","1"]))
st.plotly_chart(specific_model_hist, sharing='streamlit')

##################################################################################################
# more advanced plots

# Price hist with specific model as marker/color
bin_width = 499 #km # cannot decide on a bin size, but can decide on the number of bins (and we can compute how many bins we need for a specific bin size)
nbins = math.ceil((df["Price"].max() - df["Price"].min()) / bin_width)
price_distr = px.histogram(df, x='Price', nbins=nbins, color="specific_model", title='Price per specific model')
st.plotly_chart(price_distr, sharing='streamlit')

# Scatterplot Mileage vs Price (with specific model as marker)
scatter_price_mileage = px.scatter(df, 
    x='Mileage', y='Price', 
    title='Scatterplot Mileage vs Price', 
    trendline='ols',                    #regression line but requires statsmodels to be installed
    color='specific_model',
    height = 600, width=800) 
st.plotly_chart(scatter_price_mileage, sharing='streamlit')

