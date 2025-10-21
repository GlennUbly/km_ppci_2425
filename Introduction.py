# Streamlit app for PPCI travel times in Kent and Medway
# 13/10/2025 GU
# Page 0 for intro
# Updated 13/10/2025 for use of 24/25 base data

import streamlit as st

st.title('Reducing Travel Times to Treatment for PPCI Patients in Kent and Medway ICB - 2024/25 update')

st.markdown('This application is based on work originally created as as part of the [HSMA4 programme](https://sites.google.com/nihr.ac.uk/hsma).')

st.markdown("#### A comparison of possible configurations of sites offering "+
            "this service, looking at the potential impact on travel times "+
            "and distances")

st.markdown('This is an update of earlier work based on the [2022/23 activity](https://kmppci.streamlit.app/) ')

st.markdown('Please use the sidebar on the left to navigate')

with st.expander('Resources used:',
                 expanded=False) :
    st.markdown('This is app is created and shared using using [Streamlit](https://streamlit.io/) ')
    st.markdown('Time and distance calculations from [Routino](https://www.routino.org/uk/) ')
    st.markdown('NHS activity data from [National Commissioning Data Repository](https://www.ardengemcsu.nhs.uk/services/business-intelligence/ncdr/) ')
    st.markdown('LSOA geographic data from [Cambridgeshire Insight Open Data](https://data.cambridgeshireinsight.org.uk/dataset/output-areas)')
    st.markdown('NHS ICB geographic data from [Office for National Statistics](https://geoportal.statistics.gov.uk/maps/d6acd30ad71f4e14b4de808e58d9bc4c)')
    st.markdown('##### Python dependencies:')
    st.text("""
            bokeh==2.4.3
            geopandas==0.9.0
            matplotlib==3.5.1
            numpy==1.22.3
            pandas<2.0
            Pillow==9.3.0
            scipy==1.11.4
            seaborn==0.11.2
            streamlit==1.31.1
            """)


