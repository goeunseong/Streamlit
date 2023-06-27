import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro
from sklearn.utils import resample
from AB_test import test   
from prep_data import pre_data
from load_data import load_csv
import streamlit as st

st.title('A/B Test for creative optimization')
st.write()

# DATA_URL = st.text_input(label='URL', value='기본값', help='파일이 저장된 URL을 입력하세요.')
menu = ['CSV']
choice = st.sidebar.selectbox('메뉴',menu)
upload_csv = st.file_uploader('CSV 파일을 선택', type=['csv'], 
                              accept_multiple_files=False)
for file in upload_csv:
    df = pd.read_csv(file, encoding='cp949')
    st.dataframe(df)

# # path ='https://drive.google.com/uc?id='+ DATA_URL.split('/')[-2]

# data_load_state = st.text('Loading data...')

# # @st.cache_data
# # def load_data(path):
# #     data = pd.read_csv(path, encoding='utf-8-sig', sep='\t')
# #     return data

# data = load_data(path)
# st.dataframe(data)
# # # data_load_state.text("Done! (using st.cache)")  





