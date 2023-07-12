import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils import resample
from scipy.stats import shapiro, ttest_ind, norm
import scipy.stats as stats
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import time

@st.cache_data
def load_data():
    if file is not None:
        ext = file.name.split('.')[-1]
        if ext == 'csv':
            # 파일 읽기
            df = pd.read_csv(file, encoding='euc-kr')
            # 출력
            # st.dataframe(df)
        elif 'xls' in ext:
            # 엑셀 로드
            df = pd.read_excel(file, engine='openpyxl')
            # 출력
            # st.dataframe(df)
    return df

@st.cache_data
def prep_data():
    global df
    # 그룹별 소재구분값 생성 
    label = df.groupby(['날짜', '소재명'])['소재 구분'].unique().to_frame().reset_index()
    label['소재 구분'] = label['소재 구분'].apply(lambda x: 1 if x==[1] else 0)

    # 데이터셋 전처리(날짜, 소재명을 기준으로 그룹화)
    df = df.groupby(['날짜', '소재명'])[['노출', '클릭', '앱 설치']].sum()
    df.reset_index(level = ['날짜', '소재명'], inplace=True)

    # 소재별 클릭률, 전환율 계산
    df['클릭률'] = round(df['클릭']/df['노출'],4).replace([np.nan, np.inf], 0)
    df['전환율'] = round(df['앱 설치']/df['클릭'],4).replace([np.nan, np.inf], 0)

    # 데이터 병합
    df = pd.merge(df, label)
    return df

def calc_sample_size():
    calculate_state = st.text('Calculating···')
    data = prep_data()
    global click_1, click_0, conv_1, conv_0
    click_1 = data[data['소재 구분']==1]['클릭률'].to_list()
    click_0 = data[data['소재 구분']==0]['클릭률'].to_list()
    conv_1 = data[data['소재 구분']==1]['전환율'].to_list()
    conv_0 = data[data['소재 구분']==0]['전환율'].to_list()

    # 집단의 평균, 표준 편차, 샘플 크기를 기준으로 샘플 크기 계산
    mean1 = np.mean(click_1)    # 첫 번째 집단의 평균
    mean2 = np.mean(click_0)    # 두 번째 집단의 평균
    std1 = np.std(click_1)      # 첫 번째 집단의 표준 편차
    std2 = np.std(click_0)      # 두 번째 집단의 표준 편차
    alpha = 0.05  # 유의수준 (보통 0.05)
    power = 0.8   # 검정력 (1-beta, 보통 0.8)

    # 표준 정규 분포에서의 Z값 계산
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    # 효과 크기 계산 (Cohen's d)
    effect_size = (mean1 - mean2) / ((std1**2 + std2**2) / 2)**0.5

    # 샘플 크기 계산
    sample_size = ((z_alpha + z_beta)**2 * (std1**2 + std2**2) / 2) / effect_size**2
    st.text(f"Recommended Sample Size : {sample_size*1000}")
    uploaded_data = st.text('')
    if len(data) < sample_size:
        uploaded_data.text(f'업로드 하신 데이터의 크기는 {len(data)}로, 검정을 위해 권장되는 표본의 크기보다 {sample_size - len(data)} 만큼 작아요!')
    else:
        uploaded_data.text(f'업로드 하신 데이터의 크기는 {len(data)}로, 검정을 수행하기에 충분해요!')
    calculate_state.text("데이터 샘플링 완료!")

def resampling():
    with st.spinner('In progress'):
        time.sleep(1)
    df= prep_data()
    global click_1, click_0, conv_1, conv_0
    click_1 = df[df['소재 구분']==1]['클릭률'].to_list()
    click_0 = df[df['소재 구분']==0]['클릭률'].to_list()
    conv_1 = df[df['소재 구분']==1]['전환율'].to_list()
    conv_0 = df[df['소재 구분']==0]['전환율'].to_list()

    if len(click_1) < len(click_0):
        click_1 = resample(click_1, replace=True, n_samples=len(click_0))
    elif len(click_1) > len(click_0):
        click_0 = resample(click_0, replace=True, n_samples=len(click_1))
    if len(conv_1) < len(conv_0):
        conv_1 = resample(conv_1, replace=True, n_samples=len(conv_0))
    elif len(conv_1) > len(conv_0):
        conv_0 = resample(conv_0, replace=True, n_samples=len(conv_1))
    
    if 'click_0' not in st.session_state:
        st.session_state.click_0 = click_0
    
    if 'click_1' not in st.session_state:
        st.session_state.click_0 = click_1
    
    if 'conv_0' not in st.session_state:
        st.session_state.click_0 = conv_0

    if 'conv_1' not in st.session_state:
        st.session_state.click_0 = conv_1

    st.success('Done!')

    return click_0, click_1, conv_0, conv_1

def test(a,b, text):
    two_s_test = stats.wilcoxon(a, b, alternative = 'two-sided')
    if two_s_test[1] <= 0.05:
        one_s_test = stats.wilcoxon(a, b, alternative = 'greater')
        result = f'{text}을(를) 사용한 소재와 미사용한 소재의 성과에 차이가 있어요'
        direction = f'{text}을(를) 사용한 소재가 미사용한 소재보다 성과가 뛰어나요'
    else:
        st.write('검정 결과 : {text}을/를 사용한 소재와 미사용한 소재의 성과에 차이가 없어요')
    
    if one_s_test[1] >= 0.05:
        one_s_test = stats.wilcoxon(a,b, alternative = 'less')
        direction = '모델을(를) 사용한 소재가 모델을 미사용한 소재보다 성과가 저조해요'
    else:
        pass
    
    st.write('<p style="font-family:sans-serif; font-size:15px; color:blue;">양측 검정 결과  ', f' ▷ {result}</p>', unsafe_allow_html=True) 
    st.write(two_s_test)
    st.write('<p style="font-family:sans-serif; font-size:15px; color:blue;">단측 검정 결과  ', f' ▷ {direction}</p>', unsafe_allow_html=True) 
    st.write(one_s_test)

with st.sidebar:
    choose = option_menu("AB_TEST", ["About", "A/B Test", "Update log"],
                         icons=['house', 'cpu fill', 'kanban'],
                         menu_icon="app-indicator", default_index=1,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose == 'A/B Test':
    # 앱 메인 타이틀
    st.title('A/B Test for creative optimization')
    # 파일 업로드 버튼(업로드 기능)
    file = st.file_uploader('파일 선택(csv or excel)', type=['csv', 'xls', 'xlsx'])
    if file is not None:
        df = load_data()
        st.dataframe(df)
        prep = st.button("일자 & 소재별로 클릭률 · 전환율을 알고 싶다면? 👆")
        if prep:
            df = prep_data()
            st.dataframe(df)
        option = st.selectbox("A/B Test 이전 수행할 항목을 선택해주세요!", ('선택', '샘플 크기 추정','데이터 리샘플링'))
        if option == '샘플 크기 추정':
            calc_sample_size()
        elif option == '데이터 리샘플링':
            click_0, click_1, conv_0, conv_1 = resampling()
            
        option_2 = st.selectbox('수행할 A/B 테스트 방법을 선택해주세요', ('선택', 'Paired T-test', 'One sample T-test', 'Two sample T-test'))
        st.warning("""Paired T-test는 실험·대조군의 샘플 크기가 동일해야 가능해요.  
                   크기가 다른 경우, 데이터 리샘플링을 실행해 주세요!""")
        option_3 = st.radio('테스트할 지표를 선택해주세요', ('선택', '클릭률', '전환율'))
        if option_2 == 'Paired T-test':
            if option_3 == '클릭률':
                prog = st.progress(0, text = 'wait for it...')
                for pr in range(100):
                    time.sleep(0.02)
                    prog.progress(pr+1, text = 'wait for it...')
                test(click_0, click_1, '모델')

        
elif choose == 'About':
    st.markdown("""안녕하세요, :blue[A/B Test 사이트]입니다:sunglasses:. 왼쪽 메뉴탭의 **A/B Test**에서  
                상단의 Browse files를 클릭하여 A/B Test를 진행하고자 하는 파일을 업로드 해주세요!  
                이후 생성되는 가이드에 따라 A/B Test를 진행하시면 됩니다:)""")
elif choose == 'Update log':
    st.write('<p style="font-size:15px; color:green;">1차 업데이트 완료_샘플 크기 추정(on 2023.07.09)</p>', unsafe_allow_html=True)
    st.write('<p style="font-size:15px; color:green;">2차 업데이트 완료_데이터 리샘플링(on 2023.07.09)</p>', unsafe_allow_html=True)
    st.write('<p style="font-size:15px; color:blue;">3차 업데이트 완료_가설 검정 모델링 및 테스트(on 2023.07.09)</p>', unsafe_allow_html=True)