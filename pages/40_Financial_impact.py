# Streamlit app for PPCI travel times in Kent and Medway
# 13/10/2025 GU
# Page 5 for calculations on activity and cost
# for possible new site configurations, 18/02/2026

##############################################################################
#
#                  Package imports and initial data sources
#
##############################################################################

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from bokeh.plotting import ColumnDataSource, figure, output_file, show, output_notebook
from bokeh.models import LinearInterpolator
from shapely import wkt
from shapely.geometry import Point
import seaborn as sbn
import numpy as np
import itertools
import streamlit as st
import time


##############################################################################
#
#                       Functions for processing  data
#
##############################################################################

# Function to return the dataframe of times and distances nationally
# Input is the national activity file and the Routino output
st.cache_data()
def get_activity_time_dist_df(filename_activity, filename_routino):
    activity_df = pd.read_csv(filename_activity)
    
    # Fix anomalous postcode in actuals, and add in trimmed postcode
    activity_df['Provider_Site_Postcode'] = activity_df['Provider_Site_Postcode'].replace('CB2 0AY','CB2 0AA')
    activity_df['Site_Postcode_trim'] = activity_df['Provider_Site_Postcode'].str.replace(' ','')
    activity_df['Activity'] = 1    
    
    # Read Routino output file on national activity
    filename_routino = 'actuals_from_to_routino.csv'
    routino_output = pd.read_csv(filename_routino)
    
    # Filter and tidy up Routino output
    nat_from_to_df = routino_output[['from_postcode','to_postcode','time_min','distance_km']].drop_duplicates()
    nat_from_to_df['to_postcode'] = nat_from_to_df['to_postcode'].str.replace(' ','')
    
    # Merge time and distance into the activity DataFrame
    activity_time_dist_df = pd.merge(activity_df, 
                                     nat_from_to_df, 
                                     how='left', 
                                     left_on=['Patient_LSOA','Site_Postcode_trim'],
                                     right_on=['from_postcode','to_postcode']
                                    )
    
    return activity_time_dist_df[['distance_km','time_min']]


# Function to return national mean travel times by ICB from national activity file and Routino output
st.cache_data()
def get_national_activity_icb(filename_activity, filename_routino):
    activity_df = pd.read_csv(filename_activity)
    
    # Fix anomalous postcode in actuals, and add in trimmed postcode
    activity_df['Provider_Site_Postcode'] = activity_df['Provider_Site_Postcode'].replace('CB2 0AY','CB2 0AA')
    activity_df['Site_Postcode_trim'] = activity_df['Provider_Site_Postcode'].str.replace(' ','')
    activity_df['Activity'] = 1    
    
    # Read Routino output file on national activity
    filename_routino = 'actuals_from_to_routino.csv'
    routino_output = pd.read_csv(filename_routino)
    
    # Filter and tidy up Routino output
    nat_from_to_df = routino_output[['from_postcode','to_postcode','time_min','distance_km']].drop_duplicates()
    nat_from_to_df['to_postcode'] = nat_from_to_df['to_postcode'].str.replace(' ','')
    
    # Merge time and distance into the activity DataFrame
    activity_time_dist_df = pd.merge(activity_df, 
                                     nat_from_to_df, 
                                     how='left', 
                                     left_on=['Patient_LSOA','Site_Postcode_trim'],
                                     right_on=['from_postcode','to_postcode']
                                    )
    
    # Calculate mean travel times by ICB
    icb_time_df = (activity_time_dist_df[['Pt_ICB_Code','time_min']]
                   .groupby('Pt_ICB_Code')
                   .agg({'time_min':'mean'})
                   .reset_index()
                   .sort_values(by='time_min',ascending=False)
                  )
    
    return icb_time_df


# Function to return national mean travel times by ICB from national activity file and Routino output
st.cache_data()
def get_national_activity_prov(filename):
    activity_df = pd.read_csv(filename)
    activity_df['Activity'] = 1
    # Calculate total activity by provider site
    prov_activity_df = (activity_df[['Provider_Site_Code','Activity']]
                        .groupby('Provider_Site_Code')
                        .agg({'Activity':'sum'})
                        .reset_index()
                        .sort_values(by='Activity',ascending=False)
                       )
    return prov_activity_df


# Read zipfile containing ICB shapefile and .csv with population values, and merge for expanded shapefile
st.cache_data()
def get_icb_gdf(filename_geo, filename_pop):
    
    # Read ICB shapefile
    icb_gdf = gpd.read_file(filename_geo, crs='EPSG:27700')
    icb_gdf['ICB22NM'] = icb_gdf['ICB22NM'].str.replace('Integrated Care Board', 'ICB')
    
    # Read ICB population file
    icb_pop = pd.read_csv(filename_pop)
    icb_pop['Integrated Care Board'] = icb_pop['Integrated Care Board'].str.replace(' Of ',' of ')
    icb_pop['Integrated Care Board'] = icb_pop['Integrated Care Board'].str.replace(' The ',' the ')
    
    # Merge in population values
    icb_gdf = icb_gdf.merge(icb_pop, left_on='ICB22NM', right_on='Integrated Care Board')
    
    # Select columns and rename
    cols_to_keep = ['ICB22','ICB22NM','geometry','Projected population 2022/23','Region']
    icb_gdf = icb_gdf[cols_to_keep]
    icb_gdf.rename(columns={'ICB22':'ICB_code','ICB22NM':'ICB_name','Projected population 2022/23':'Population'}, inplace=True)
    
    # Calcualte area and population density for each ICB from the (multi)polygon
    icb_gdf['Area'] = icb_gdf['geometry'].area
    icb_gdf['Population_per_sq_km'] = 10**6 * icb_gdf['Population'] / icb_gdf['Area']
    
    return icb_gdf


# Function to return the GeoDataFrame with the provider site locations
st.cache_data()
def get_prov_gdf(filename):
    df = pd.read_csv(filename)
    gdf_prov = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.Longitude_1m, df.Latitude_1m))
    gdf_prov = gdf_prov.set_crs(epsg=4326)
    gdf_prov = gdf_prov.to_crs(epsg=27700)
    return gdf_prov


# GDF to plot with ICB mean travel times and providers with activity above a threshold
# Input is: 
#   the national_activity_prov DataFrame grouped by provider
#   the icb_time_df with mean travel times for each ICB
#   the icb_gdf GeoDataFrame for plotting the national map by OCB
#   the prov_gdf GeoDataFrame for plotting the locations of the provider sites
#   the threshold for including a provider in the map, default 50
st.cache_data()
def national_activity_to_plot(national_activity_prov, icb_time_df, icb_gdf, prov_gdf, threshold=50):
    # Filter the providers to exclude those with minimal activity
    sites_to_include = national_activity_prov[national_activity_prov['Activity'] > threshold].index
    prov_gdf_include = prov_gdf[prov_gdf['Der_Provider_Site_Code'].isin(sites_to_include)]
    
    # Merge icb_gdf with icb_time_df for ICB plot
    national_icb_times = icb_gdf.merge(icb_time_df,
                                       how='left',
                                       left_on=['ICB_code'],
                                       right_on=['Pt_ICB_Code'],
                                      )
    
    return (prov_gdf_include, national_icb_times)


# Function to plot map with ICB mean traval times and providers with activity above a threshold
# Inputs are the GeoDataFrames from national_activity_to_plot for the ICBs by travel time, and the provider site locations
st.cache_data()
def plot_national_icb_prov(prov_to_plot, icb_to_plot):
    # Figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot mean travel times for each ICB, colour from travel time
    icb_to_plot.plot(ax=ax,
                     column='time_min',
                     edgecolor='green',
                     linewidth=0.5,
                     vmin=0,
                     vmax=110,
                     cmap='inferno_r',
                     legend_kwds={'shrink':0.8, 'label':'Travel time (mins)'},
                     legend=True,
                     alpha = 0.70)
    
    # Plot location of hospitals (filtered to activity > 50)
    prov_to_plot.plot(ax=ax, 
                          edgecolor='k', 
                          facecolor='w', 
                          markersize=200,
                          marker='*')
    
    # Axis details
    ax.set_axis_off()
    ax.set_title('PPCI Mean Travel Time by ICB')
    ax.margins(0)
    ax.apply_aspect()
    plt.subplots_adjust(left=0.01, right=1.0, bottom=0.0, top=1.0)
    
    # Save figure
    plt.savefig('national_map.jpg', dpi=300)
    
    return fig, ax


# Function to return national median travel time
st.cache_data()
def get_national_median(filename):
    activity_df = pd.read_csv(filename)
    nat_median = activity_df['time_min'].median()
    return nat_median


# Function to return DataFrame for population density/travel time correlation scatter plot
# Input is the ICB element of the national plot used above
st.cache_data()
def get_density_times_icb_rank_df(icb_density_times_df):
    density_times_icb_rank_df = icb_density_times_df.copy()
    density_times_icb_rank_df['Travel_time_rank_asc'] = density_times_icb_rank_df['time_min'].rank(ascending=True)
    density_times_icb_rank_df['Pop_density_rank'] = density_times_icb_rank_df['Population_per_sq_km'].rank(ascending=False)
    return density_times_icb_rank_df


# Function to return Bokeh correlation chart of population density and travel times
# Input is the icb_density_times_df DataFrame calcuated by get_density_times_icb_df above
st.cache_data()
def plot_density_times(density_times_icb_gdf):
    density_times_icb_gdf['Travel_time_rank_asc'] = density_times_icb_gdf['time_min'].rank(ascending=True)
    density_times_icb_gdf['Pop_density_rank'] = density_times_icb_gdf['Population_per_sq_km'].rank(ascending=False)
    corr_df = density_times_icb_gdf[['ICB_name',
                                     'time_min',
                                     'Travel_time_rank_asc',
                                     'Pop_density_rank',
                                     'Population_per_sq_km',
                                     'Population'
                                    ]]

    size_mapper=LinearInterpolator(
        x=[corr_df.Population.min(),corr_df.Population.max()],
        y=[5,20]
        )

    source = ColumnDataSource(data=dict(
        x=corr_df['Population_per_sq_km'],
        y=corr_df['time_min'],
        desc1=corr_df['ICB_name'],
        desc2=corr_df['Population'],
        ))

    TOOLTIPS = [
        ("ICB", "@desc1"),
        ("Population density", "@x{0}"),
        ("Mean travel time", "@y{0}"),
        ("Population", "@desc2{0}")
        ]

    fig = figure(width=600,
                 height=600,
                 tooltips=TOOLTIPS,
                 title="Population density against travel time by ICB")

    fig.circle('x',
               'y',
               size={'field':'desc2','transform': size_mapper},
               source=source)
    fig.xaxis.axis_label = 'Population density (per sq km)'
    fig.yaxis.axis_label = 'Mean travel time (mins)'
    return fig


##############################################################################
#
#        Functions for processing data for Kent and Medway journeys
#
##############################################################################

# Function to return the GeoDataFrame for the LSOA map and calculations
st.cache_data()
def get_lsoa_gdf(filename):
    km_lsoa_df = pd.read_csv(filename)
    km_lsoa_df['geometry'] = km_lsoa_df['geometry'].apply(wkt.loads)
    km_lsoa_gdf = gpd.GeoDataFrame(km_lsoa_df, crs='epsg:27700')
    return km_lsoa_gdf


# Function to return the map of potential sites
# Input is the km_lsoa_gdf of all K&M LSOAs, and the prov_gdf of the provider sites
st.cache_data()
def get_map_of_sites(km_lsoa_gdf, km_sites_gdf):
    fig, ax = plt.subplots(figsize=(10, 20))
    km_lsoa_gdf.plot(ax=ax)
    km_sites_gdf.plot(ax=ax, edgecolor='k', facecolor='gold', markersize=200,marker='*')
    for x, y, label in zip(km_sites_gdf.geometry.x, 
                           km_sites_gdf.geometry.y, 
                           km_sites_gdf.Provider_Site_Name):
        ax.annotate(label, xy=(x, y), 
                    xytext=(8, 8), 
                    textcoords="offset points",
                    backgroundcolor="y",
                    fontsize=8)
    axis_title = 'Current and potential PPCI sites for Kent and Medway \n'
    ax.set_axis_off()
    ax.set_title(axis_title, fontsize=16)
    return fig, ax


# KDE plot for current travel times in K&M with comparison to national median
# Input is the km_lsoa_gdf of all K&M LSOAs, the current provider list, the proposed new sites, and the threshold value
# Use for K&M intro on the current position, and for the additional site plots
st.cache_data()
def kde_plot(gdf, sites_orig, sites_new, nat_median):
    cols = ['lsoa11cd','geometry']
    cols_orig = ['time_'+site.lower() for site in sites_orig]
    cols_new = ['time_'+site.lower() for site in sites_new]
    cols.extend(['time_min'] + cols_orig + cols_new)
    gdf_calculated_min_time = gdf[cols].copy()
    gdf_calculated_min_time['orig_min_time'] = gdf_calculated_min_time[cols_orig].min(axis=1)
    gdf_calculated_min_time['new_min_time'] = gdf_calculated_min_time[cols_orig + cols_new].min(axis=1)
    gdf_calculated_min_time['orig_<_nat'] = np.where(gdf_calculated_min_time['orig_min_time'] < nat_median , True , False)
    gdf_calculated_min_time['Compare_with_national'] = np.where(gdf_calculated_min_time['orig_<_nat'],
                                                                'Less than or equal to national median',
                                                                'Greater than national median')
    sites = [dict_sitecode_sitename[code] for code in sites_new]
    fig, ax = plt.subplots(figsize=(10,6))
    sbn.kdeplot(ax=ax,
                data=gdf_calculated_min_time['orig_min_time'],
                clip = (0,125),
                fill=True,
                legend= True
               )
    plt.axvline(x=gdf_calculated_min_time['time_min'].median(),
                linewidth=2,
                color='cornflowerblue')
    
    if len(sites_new) > 0:
        sbn.kdeplot(ax=ax,
                    data=gdf_calculated_min_time['new_min_time'],
                    clip = (0,125),
                    fill=True,
                    legend= True
                   ).set(title='Travel times with additional site at:\n '+
                                 '\n'.join(sites))
        plt.axvline(x=gdf_calculated_min_time['new_min_time'].median(),
                    linewidth=2,
                    color='orange')
        plt.axvline(x=nat_median,
                linewidth=2,
                color='green')
        ax.set_xlabel('Travel time (mins)')
        plt.legend(('Travel times current sites',
                    'Median current travel time',
                    'Travel times with additional sites',
                    'Median with additional sites',
                    'National median travel time'),
                   loc='upper right',fontsize=8)
        
    else:
        plt.axvline(x=nat_median,
                linewidth=2,
                color='green')
        plt.title('Current Kent and Medway PPCI travel times')
        plt.legend(('Travel times current sites',
                    'Median current travel time',
                    'National median travel time'),
                   loc='upper right',fontsize=8)
    return fig, ax


# Function to get dictionary mapping site codes to site names
st.cache_data()
def get_dict_sitecode_sitename(sites_df):
    dic = dict(zip(sites_df.Provider_Site_Code, sites_df.Provider_Site_Name))
    return dic

# Function to get dictionary mapping site codes to site postcodes
def get_dict_sitecode_postcode(sites_df):
    dic = dict(zip(sites_df.Provider_Site_Code, sites_df.Postcode_Trim))
    return dic

# Function to get dictionary mapping site names to site codes
def get_dict_sitename_sitecode(sites_df):
    dic = dict(zip(sites_df.Provider_Site_Name, sites_df.Provider_Site_Code))
    return dic

# Function to get dictionary mapping site postcodes to site codes
def get_dict_postcode_sitecode(sites_df):
    dic = dict(zip(sites_df.Postcode_Trim, sites_df.Provider_Site_Code))
    return dic

# Function to return a new LSOA GDF with calculated minimum times for a new site list and threshold comparison
# Input is the LSOA GeoDataFrame with all times, the list of original sites, the list of new sites, and the national median
st.cache_data()
def get_new_min_times_gdf(km_lsoa_gdf, sites_orig, sites_new, threshold):
    cols = ['lsoa11cd','geometry']
    cols_orig = ['time_'+a.lower() for a in sites_orig]
    cols_new = ['time_'+a.lower() for a in sites_new]
    cols.extend(cols_orig + cols_new)
    gdf_calculated_min_time = km_lsoa_gdf[cols].copy()
    gdf_calculated_min_time['orig_min_time'] = gdf_calculated_min_time[cols_orig].min(axis=1)
    gdf_calculated_min_time['new_min_time'] = gdf_calculated_min_time[cols_orig + cols_new].min(axis=1)
    gdf_calculated_min_time['impact_time'] = gdf_calculated_min_time['new_min_time'] - gdf_calculated_min_time['orig_min_time']
    gdf_calculated_min_time['orig_<_nat'] = np.where(gdf_calculated_min_time['orig_min_time'] < nat_median , True , False)
    gdf_calculated_min_time['new_<_nat'] = np.where(gdf_calculated_min_time['new_min_time'] < nat_median , True , False)
    gdf_calculated_min_time['Compare_with_national'] = np.where(gdf_calculated_min_time['orig_<_nat'],
                                                                'Remains <= national median',
                                                                np.where(gdf_calculated_min_time['new_<_nat'],
                                                                         'Change to <= national median',
                                                                'Remains > national median')
                                                               )
    # New columns to show 
    site_time_new_list = list(gdf_calculated_min_time.columns)[2:2+len(sites_new)+len(sites_orig)]
    site_time_orig_list = list(gdf_calculated_min_time.columns)[2:2+len(sites_orig)]
    gdf_calculated_min_time['site_of_new_time'] = gdf_calculated_min_time[site_time_new_list].idxmin(axis=1)
    gdf_calculated_min_time['closest_site_current'] = gdf_calculated_min_time[site_time_orig_list].idxmin(axis=1)
    return gdf_calculated_min_time

# Function to plot the new GDF with the minimum time for the existing and new sites 
# with colour for each LSOA representing travel time
# Input is the LSOA GeoDataFrame with all times, the list of original sites, the list of new sites, and the national median
st.cache_data()
def plot_lsoa_times_gdf(km_lsoa_gdf, sites_orig, sites_new, threshold):
    new_min_times_gdf = get_new_min_times_gdf(km_lsoa_gdf, sites_orig, sites_new, threshold)
    sites = [dict_sitecode_sitename[code] for code in sites_new]
    # Plot the LSOAs with colour for travel time
    fig, ax = plt.subplots(figsize=(10,6))
    new_min_times_gdf.plot(ax=ax,
                           column='new_min_time',
                           legend=True,
                           legend_kwds={'shrink':0.8, 'label':'Travel time (mins)'},
                           #cax=cax
                          )
    # And plot the site locations
    test_sites_gdf = km_prov_gdf[km_prov_gdf['Provider_Site_Code'].isin(sites_orig + sites_new)]
    test_sites_gdf.plot(ax=ax,
                        edgecolor='r',
                        facecolor='silver',
                        markersize=200,
                        marker='*')
    ax.set_axis_off()
    if len(sites_new) == 0:
        ax.set_title('Travel times for current sites',fontsize=16)
    else:
        ax.set_title('Travel times with the addition of sites:\n '+
                     '\n'.join(sites),
                     fontsize=16)
    return fig, ax


# Function to show impact of new sites on travel times, with the change in travel times minutes
# with colour for each LSOA representing the reduction in travel time
# Input is the LSOA GeoDataFrame with all times, the list of original sites, the list of new sites, and the national median
#st.cache_data()
def plot_lsoa_times_impact_gdf(km_lsoa_gdf, sites_orig, sites_new, threshold):
    new_min_times_gdf = get_new_min_times_gdf(km_lsoa_gdf, sites_orig, sites_new, threshold)
    sites = [dict_sitecode_sitename[code] for code in sites_new]
    # Plot the LSOAs with colour for travel time
    fig, ax = plt.subplots(figsize=(10,6))
    new_min_times_gdf.plot(ax=ax,
                           column='impact_time',
                           legend=True,
                           legend_kwds={'shrink':0.8, 'label':'Travel time impact (mins)'},
                           #cax=cax
                          )
    # And plot the site locations
    test_sites_gdf = km_prov_gdf[km_prov_gdf['Provider_Site_Code'].isin(sites_orig + sites_new)]
    test_sites_gdf.plot(ax=ax,
                        edgecolor='r',
                        facecolor='silver',
                        markersize=200,
                        marker='*')
    ax.set_axis_off()
    if len(sites_new) == 0:
        ax.set_title('No additional sites selected',fontsize=16)
    else:
        ax.set_title('Impact on travel times with the addition of sites:\n '+
                     '\n'.join(sites),
                     fontsize=16)
    return fig, ax


# Function to plot the new GDF with the minimum time for the existing and new sites 
# with colour for each LSOA representing whether the travel time is greater or less than a threshold
# Input is the LSOA GeoDataFrame with all times, the list of original sites, the list of new sites, and the national median
#st.cache_data()
def plot_lsoa_time_threshold_gdf(km_lsoa_gdf, sites_orig, sites_new, threshold):
    sites = [dict_sitecode_sitename[code] for code in sites_new]
    new_min_times_gdf = get_new_min_times_gdf(km_lsoa_gdf, sites_orig, sites_new, threshold)
    # Plot the LSOAs with colour for travel time
    fig, ax = plt.subplots(figsize=(10,6))
    new_min_times_gdf.plot(ax=ax,
                           column='Compare_with_national',
                           legend=True,
                           #legend_kwds={'shrink':0.8, 'label':'Travel time (mins)'},
                           #cax=cax
                          )
    # And plot the site locations
    test_sites_gdf = km_prov_gdf[km_prov_gdf['Provider_Site_Code'].isin(sites_orig + sites_new)]
    test_sites_gdf.plot(ax=ax,
                        edgecolor='r',
                        facecolor='silver',
                        markersize=200,
                        marker='*')
    ax.set_axis_off()
    if len(sites_new) == 0:
        ax.set_title('Travel times for current sites compared to national median',fontsize=16)
    else:
        ax.set_title('Impact on travel time compared to national median for additional sites\n '+
                     '\n'.join(sites),
                     fontsize=16)
    return fig, ax


# Function to return KM activity from national actuals
# Input is the national actuals file, and the national Routino output on the actuals
#st.cache_data()
def get_km_actuals_time_dist_df(filename_activity, filename_routino):
    # Create df for KM actuals with only from and to columns
    activity_df = pd.read_csv(filename_activity)
    km_activity_df = activity_df[activity_df['Pt_ICB_Code']=='QKS'].copy()
    km_activity_df = km_activity_df[['Provider_Site_Postcode','Patient_LSOA']]
    km_activity_df['Provider_Site_Postcode'].replace('CB2 0AY','CB2 0AA',inplace=True)
    km_activity_df['Provider_Site_Postcode'] = km_activity_df['Provider_Site_Postcode'].str.replace(' ','')
    # Create Routino table for merging with the actuals
    routino_output = pd.read_csv(filename_routino)
    nat_from_to_df = routino_output[['from_postcode','to_postcode','time_min','distance_km']].drop_duplicates()
    nat_from_to_df['to_postcode'] = nat_from_to_df['to_postcode'].str.replace(' ','')
    # Merge time and distance into the activity DataFrame
    km_actuals_time_dist_df = pd.merge(km_activity_df, 
                                       nat_from_to_df, 
                                       how='left', 
                                       left_on=['Patient_LSOA','Provider_Site_Postcode'],
                                       right_on=['from_postcode','to_postcode']
                                      )
    return km_actuals_time_dist_df


# Function to return all potential journeys for each actual record
# Input is the km_actuals_time_dist_df dataframe of actual from-to items, 
# and the km_lsoa_gdf KM Routino table of all potential from/to times and distances for KM patients
#st.cache_data()
def get_km_all_journeys_df(km_actuals_time_dist_df, km_lsoa_gdf):
    km_all_journeys_df = pd.merge(km_actuals_time_dist_df,
                                  km_lsoa_gdf,
                                  how='left',
                                  left_on=['from_postcode'],
                                  right_on=['lsoa11cd']        
                                 ).drop(columns=['Provider_Site_Postcode',
                                                 'Patient_LSOA','objectid',
                                                 'lsoa11nm'])
    return km_all_journeys_df

# Function to return a table of activity, showing 24/25 actuals, and for these actuals the closest sites under the 
# present configuration, and the possible activity if we add new sites
# Input all_journeys_df is the actuals
def get_freq_table_actuals(all_journeys_df, sites_orig, sites_new, km_prov_gdf, threshold):
    sites_to_include = sites_orig + sites_new
    dict_sitecode_postcode = get_dict_sitecode_postcode(km_prov_gdf)
    dict_postcode_sitecode = get_dict_postcode_sitecode(km_prov_gdf)
    total_count = len(all_journeys_df)
    postcodes_to_include = [dict_sitecode_postcode[code] for code in sites_orig]
    # series for actuals
    val_count_actuals = all_journeys_df['Provider_Site_Postcode'][all_journeys_df['Provider_Site_Postcode'].isin(postcodes_to_include)].value_counts()
    val_count_actuals = val_count_actuals.to_frame().reset_index()
    val_count_actuals.columns = ['Provider_code', 'Actual_activity_count']
    val_count_actuals['Provider_code'] = val_count_actuals['Provider_code'].apply(lambda code: dict_postcode_sitecode[code])
    val_count_actuals.set_index('Provider_code', inplace=True)
    
    # Get new minimum times for the selected sites
    new_min_times_gdf = get_new_min_times_gdf(km_lsoa_gdf, sites_orig, sites_new, threshold)
    
    # series for closest with current sites
    km_times_all_lsoa_df = km_actuals_time_dist_df['Patient_LSOA'].to_frame().merge(new_min_times_gdf,
                                              how='left',
                                              left_on='Patient_LSOA',
                                              right_on='lsoa11cd'                                              
                                             )
    
    val_count_current_sites = km_times_all_lsoa_df['closest_site_current'].value_counts()
    val_count_current_sites = val_count_current_sites.to_frame().reset_index()
    val_count_current_sites.columns = ['Provider_code', 'Closest_site_current_on_24/25_activity']
    val_count_current_sites['Provider_code'] = val_count_current_sites['Provider_code'].apply(lambda code: code[5:].upper())
    val_count_current_sites.set_index('Provider_code', inplace=True)
    
    # series for closest with new sites
    val_count_new_sites = km_times_all_lsoa_df['site_of_new_time'].value_counts()
    val_count_new_sites = val_count_new_sites.to_frame().reset_index()
    val_count_new_sites.columns = ['Provider_code', 'Closest_site_with_new_options']
    val_count_new_sites['Provider_code'] = val_count_new_sites['Provider_code'].apply(lambda code: code[5:].upper())
    val_count_new_sites.set_index('Provider_code', inplace=True)
    
    # Combine these 3 dataframes, use merge
    df = val_count_actuals.merge(val_count_current_sites, how='outer', left_index=True, right_index=True)
    df = df.merge(val_count_new_sites, how='outer', left_index=True, right_index=True)
    
    # Add value for other sites
    df.loc['Other'] = total_count - df.sum() 
    # Replace NaN with 0 and convert to integers
    df = df.fillna(0).astype(int)
    # Order decreasing by new activity values
    df.sort_values(by='Closest_site_current_on_24/25_activity', ascending=False, inplace=True)
    return df



# Function to return metrics for a given list of candidate sites
# Input is the DataFrame km_all_journeys_df and a list of potential sites and a threshold value (national median time)
#st.cache_data()
def get_metrics_one_config_df(km_all_journeys_df, sites_orig, sites_new, nat_median):
    cols_orig_time = ['time_'+site.lower() for site in sites_orig]
    cols_orig_distance = ['distance_'+site.lower() for site in sites_orig]
    cols_new_time = ['time_'+site.lower() for site in sites_new]
    cols_new_distance = ['distance_'+site.lower() for site in sites_new]
    # Series of new minimum times
    new_min_time = km_all_journeys_df[cols_orig_time + cols_new_time].min(axis=1)
    # Series of new minimum distances
    new_min_dist = km_all_journeys_df[cols_orig_distance + cols_new_distance].min(axis=1)
    # Calculate journey times reduction
    time_reduction = new_min_time - km_all_journeys_df['time_min']
    # Count journey times reduced
    count_time_reduced = len(time_reduction[time_reduction < 0])
    # Calculate proportion of times reduced
    prop_time_reduced = count_time_reduced / len(time_reduction)
    # Count journey times less than national median
    count_time_below_nat_median = len(new_min_dist[new_min_dist < nat_median])
    # Calculate proportion of times below national median
    prop_time_below_nat_median = count_time_below_nat_median / len(new_min_dist)
    # Calculate original mean time
    original_time_mean = km_all_journeys_df['time_min'].mean()
    # Calculate original median time
    original_time_median = km_all_journeys_df['time_min'].median()
    # Calculate new mean time
    new_time_mean = new_min_time.mean()
    # Calculate new median time
    new_time_median = new_min_time.median()
    # Calculate new maximum time
    new_time_maximum = new_min_time.max()
    # Calculate new time variance
    new_time_variance = new_min_time.var()
    # Calculate new time standard deviation
    new_time_std = new_min_time.std()
    # Calculate total time reduction (as positive number)
    total_time_reduction = -time_reduction.sum()
    # Calculate time reduction per spell
    time_reduction_per_spell = total_time_reduction / len(time_reduction)
    # Calculate total distance reduction (as positive number)
    total_distance_reduction = (km_all_journeys_df['distance_km'] - new_min_dist).sum()
    # Bring the above values together in a dataframe
    df_metrics_single = pd.DataFrame({"Added_sites": [','.join(sites_new)],
                                      "Number_spells_reduced_time": [count_time_reduced],
                                      "Proportion_spells_reduced_time": [prop_time_reduced],
                                      "Number_spells_under_natl_median": [count_time_below_nat_median],
                                      "Proportion_spells_under_natl_median": [prop_time_below_nat_median],
                                      "Original_mean_time": [original_time_mean],
                                      "Original_median_time": [original_time_median],
                                      "New_mean_time": [new_time_mean],
                                      "New_median_time": [new_time_median],
                                      "New_maximum_time": [new_time_maximum],
                                      "New_time_variance": [new_time_variance],
                                      "New_time_sd": [new_time_std],
                                      "Total_time_reduction": [total_time_reduction],
                                      "Time_reduction_per_spell": [time_reduction_per_spell],
                                      "Total_distance_reduction": [total_distance_reduction]
                                     })
    df_metrics_single.set_index("Added_sites", inplace=True)
    return df_metrics_single


# Function to return a list of possible new pairs of sites to add
# Input is the list of all possible KM sites, and the list of current sites (to remove from consideration as new)
#st.cache_data()
def get_site_pairs(km_site_list, sites_orig) :
    proposed_sites = list(set(km_site_list) - set(sites_orig))
    pairs = list(itertools.product(*[proposed_sites,proposed_sites]))
    pairs_mixed = [list(i) for i in pairs if list(i)[0] != list(i)[1]]
    pairs_sorted = [sorted(i) for i in pairs_mixed]
    pairs_as_set_of_tuples = {tuple(i) for i in pairs_sorted}
    pairs_list_as_tuples = list(pairs_as_set_of_tuples)
    pairs_list = [list(i) for i in pairs_list_as_tuples]
    return pairs_list


# Function to return a summary table with all metrics for all configurations using the above get_metrics_one_config_df
# Input is the km_all_journeys_df, and list of proposed new sites and site pairs, and list of current sites
#st.cache_data()
def get_summary_table(km_prov_gdf, km_all_journeys_df, sites_orig, nat_median):
    # Set up DataFrame to record the values for each proposed site/pair
    df_results = pd.DataFrame()
    df_results["Added_sites"] = []
    df_results["Number_spells_reduced_time"] = []
    df_results["Proportion_spells_reduced_time"] = []
    df_results["Number_spells_under_natl_median"] = []
    df_results["Proportion_spells_under_natl_median"] = []
    df_results["Original_mean_time"] = []
    df_results["Original_median_time"] = []
    df_results["New_mean_time"] = []
    df_results["New_median_time"] = []
    df_results["New_time_variance"] = []
    df_results["New_time_sd"] = []
    df_results["Total_time_reduction"] = []
    df_results["Time_reduction_per_spell"] = []
    df_results["Total_distance_reduction"] = []
    df_results.set_index("Added_sites", inplace=True)
    
    # Determine list of single sites to consider
    km_site_list = list(km_prov_gdf['Provider_Site_Code'])
    for site in sites_orig :
        km_site_list.remove(site)
    
    # Determine list of pairs of sites to consider
    site_pairs = get_site_pairs(km_site_list, sites_orig)
        
    # Calculate metrics for each pair in the list, and concat with table of results
    for pair in site_pairs :
        df_to_add = get_metrics_one_config_df(km_all_journeys_df, 
                                              sites_orig, 
                                              pair, 
                                              nat_median)
        df_results = pd.concat([df_results, df_to_add])
    
    # Calculate metrics for each single site in the list, and concat with table of results
    for site in km_site_list :
        df_to_add = get_metrics_one_config_df(km_all_journeys_df, 
                                              sites_orig, 
                                              [site], 
                                              nat_median)
        df_results = pd.concat([df_results, df_to_add])
        
    return df_results

# Function to return dictionary with keys = Provider Code and label = MFF values, input a csv
def get_mff_dict(filename_mff):
    mff_df = pd.read_csv(filename_mff)
    mff_df.reset_index(inplace=True)
    mff_dict = dict(zip(mff_df['Provider Code'], mff_df['2024/25 MFF value']))
    return mff_dict

# Function to return dataframs of mean of actual spell costs by provider, input csv (feom SQL output) and mff dictionary
def get_mean_cost_by_prov(filename_mean_cost, mff_dict):
    cost_by_prov_df = pd.read_csv(filename_mean_cost)
    cost_by_prov_df.set_index('Der_Provider_Code', inplace=True)
    cost_by_prov_df['mff'] = cost_by_prov_df.index.map(mff_dict)
    cost_by_prov_df['mean_cost_without_mff'] = cost_by_prov_df['Mean spell cost'] / cost_by_prov_df['mff']
    return cost_by_prov_df

# Function to return activity and cost sentences
def get_finance_sentences(km_actuals_time_dist_df, site_code_list, original_sites_km, london_sites,
                          activity_table, mff_dict, mean_spell_cost, mean_cost_without_mff):
    
    # List of output sentences
    output = []
    total_cost_impact = 0
    
    # Activity and cost sentences for new sites
    for site_code in site_code_list:
        new_site_activity_change = activity_table.loc[site_code]['Closest_site_with_new_options']
        new_site_total_cost = mean_cost_without_mff * mff_dict[site_code[:3]] * new_site_activity_change
        total_cost_impact += new_site_total_cost
        new_site_activity_sentence = f'Additional activity at new site {dict_sitecode_sitename[site_code]} would be {new_site_activity_change} spells.'
        new_site_cost_sentence = f'The cost of the activity at the new site {site_code} would be £{new_site_total_cost:,.0f}.'
        output.append(new_site_activity_sentence)
        output.append(new_site_cost_sentence)
    
    # Activity and cost sentences for existing K&M site(s)
    for site_code in original_sites_km:
        #existing_site_activity_change = activity_table.loc[site_code]['Actual_activity_count'] - activity_table.loc[site_code]['Closest_site_with_new_options']
        sitename = dict_sitecode_sitename[site_code]
        change = (activity_table.loc[site_code]['Closest_site_with_new_options'] - activity_table.loc[site_code]['Actual_activity_count'])
        prov_code = site_code[:3]
        change_cost = change * mean_cost_without_mff * mff_dict[prov_code]
        total_cost_impact += change_cost
        if change < 0:
            existing_site_activity_sentence = f'Decreased activity at {sitename} by {-change} spells.'
            existing_site_cost_sentence = f'Decreased cost at {sitename} by £{-change_cost:,.0f}.'
        elif change == 0:
            existing_site_activity_sentence = f'No change in activity at {sitename}.'
            existing_site_cost_sentence = f'No change in cost at {sitename}.'
        else:
            existing_site_activity_sentence = f'Increased activity at {sitename} by {change} spells.'
            existing_site_cost_sentence = f'Increased cost at {sitename} by £{change_cost:,.0f}.'
        output.append(existing_site_activity_sentence)
        output.append(existing_site_cost_sentence)
    
    # Activity and cost sentences for London providers
    london_activity_change = 0
    london_cost_change = 0
    for site_code in london_sites:
        london_activity_change += activity_table.loc[site_code]['Closest_site_with_new_options'] - activity_table.loc[site_code]['Actual_activity_count']
        london_cost_change += (activity_table.loc[site_code]['Closest_site_with_new_options'] - activity_table.loc[site_code]['Actual_activity_count']) * mean_cost_without_mff * mff_dict[site_code[:3]]
        if london_activity_change < 0:
            london_activity_sentence = f'Decreased activity at London sites by {-london_activity_change} spells.'
            london_cost_sentence = f'Decreased cost at London sites by £{-london_cost_change:,.0f}.'
        elif change == 0:
            london_activity_sentence = f'No change in activity at London sites.'
            london_cost_sentence = f'No change in cost at London sites.'
        else:
            london_activity_sentence = f'Increased activity at London sites by {london_activity_change} spells.'
            london_cost_sentence = f'Increased cost at London sites by £{london_cost_change:,.0f}.'
    total_cost_impact += london_cost_change
    output.append(london_activity_sentence)
    output.append(london_cost_sentence)

    # Activity and cost sentences for other providers
    other_activity_change = activity_table.loc['Other']['Closest_site_with_new_options'] - activity_table.loc['Other']['Actual_activity_count']
    other_cost_change = other_activity_change * mean_spell_cost
    total_cost_impact += other_cost_change
    if other_activity_change < 0:
        other_activity_sentence = f'Decreased activity at other sites by {-other_activity_change} spells.'
        other_cost_sentence = f'Decreased cost at other sites by £{-other_cost_change:,.0f}.'
    elif change == 0:
        other_activity_sentence = f'No change in activity at other sites.'
        other_cost_sentence = f'No change in cost at other sites.'
    else:
        other_activity_sentence = f'Increased activity at other sites by {change} spells.'
        other_cost_sentence = f'Increased cost at other sites by £{london_cost_change:,.0f}.'
    output.append(other_activity_sentence)
    output.append(other_cost_sentence)

    # Update final cost
    if total_cost_impact < 0:
        output.append(f'Overall cost reduction of £{-total_cost_impact:,.0f} due to the Market Forces Factor differences')
    elif total_cost_impact == 0:
        output.append('No impact on overall cost')
    else:
        output.append(f'Overall cost increase of £{total_cost_impact:,.0f} due to the Market Forces Factor differences')
    
    return output

##############################################################################
#
#               Inputs and creation of objects for output
#
##############################################################################

# Setting values
filename_activity = 'output_national_ppci_2425.csv'
filename_mean_cost = 'mean_cost_by_provider.csv'
filename_mff = 'mff_2425.csv'
filename_mean_cost = 'mean_cost_by_provider.csv'
filename_routino = 'actuals_from_to_routino.csv'
filename_routino_km = 'results_km_all_sites.csv'
km_prov_filename = 'KM_Sites_Geog.csv'
km_lsoa_filename = 'km_lsoa_shapefile.csv'

km_actuals_time_dist_df = get_km_actuals_time_dist_df(filename_activity, filename_routino_km)
mff_dict = get_mff_dict(filename_mff)
original_sites_km = ['RVV01']
site_code_list = ['RPA02','RWFTW']
london_sites = ['RJZ01','RJ122']
cost_by_prov_df = get_mean_cost_by_prov(filename_mean_cost, mff_dict)
mean_spell_cost = cost_by_prov_df['PbR_Cost_2425'].sum() / cost_by_prov_df['Spell count'].sum()
mean_cost_without_mff = cost_by_prov_df['mean_cost_without_mff'].mean()
nat_median = get_national_median(filename_routino)

original_sites_km = ['RVV01']
site_code_list = ['RPA02']
london_sites = ['RJZ01','RJ122']
sites_orig = original_sites_km + london_sites
km_prov_gdf = get_prov_gdf(km_prov_filename)
km_lsoa_gdf = get_lsoa_gdf(km_lsoa_filename)
dict_sitecode_sitename = get_dict_sitecode_sitename(km_prov_gdf)
dict_sitename_sitecode = get_dict_sitename_sitecode(km_prov_gdf)

km_site_name_list = list(km_prov_gdf['Provider_Site_Name'])
for site in [dict_sitecode_sitename[site] for site in sites_orig] :
    km_site_name_list.remove(site)




#############################################################################
#############################################################################
##
##                          Streamlit content
##
#############################################################################
#############################################################################
    
st.title("Financial impact of possible configurations")

st.markdown("#### We calculate the possible activity and financial changes "+
            "for possible new PPCI sites in Kent & Medway")
st.markdown("Select a provider site or sites from the select box in the "+
            "sidebar to consider as an option alongside the "+
            "existing site at "+
            "RVV01 William Harvey Hospital and the 2 London sites. ")

# Select box in sidebar
with st.sidebar :
    with st.form('select_sites_form') :
        selected_site_pair = st.multiselect('Select a configuration for comparison:',
                                            km_site_name_list,
                                            default=[km_site_name_list[0]],
                                            key='select_sites')
        submit_button = st.form_submit_button(label="Submit")

# Convert selected names back to codes
site_code_list = [dict_sitename_sitecode[site] for site in selected_site_pair]

# Calculate activity table
activity_table = get_freq_table_actuals(km_actuals_time_dist_df, sites_orig, site_code_list, km_prov_gdf, nat_median)
sentences = get_finance_sentences(km_actuals_time_dist_df, site_code_list, original_sites_km, london_sites,
                                  activity_table, mff_dict, mean_spell_cost, mean_cost_without_mff)

if len(site_code_list) == 1:
    st.markdown(f'With an additional site at {dict_sitecode_sitename[site_code_list[0]]}, we would expect the following changes in activity and costs compared to the current configuration:')
elif len(site_code_list) > 1:
    sites = []
    for site in site_code_list:
        sites.append(dict_sitecode_sitename[site])
    st.markdown(f'With additional sites at {", ".join(sites)}, we would expect the following changes in activity and costs compared to the current configuration:')
    

st.dataframe(activity_table)

for sentence in sentences:
    st.markdown(f'* {sentence}')

f = plot_lsoa_times_gdf(km_lsoa_gdf, sites_orig, site_code_list, nat_median)
fig, ax = f
st.pyplot(fig)


