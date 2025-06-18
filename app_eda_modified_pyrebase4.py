import streamlit as st
import pyrebase
import time
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# ---------------------
# Firebase 설정 (unchanged)
# ---------------------
firebase_config = {
    "apiKey": "AIzaSyCswFmrOGU3FyLYxwbNPTp7hvQxLfTPIZw",
    "authDomain": "sw-projects-49798.firebaseapp.com",
    "databaseURL": "https://sw-projects-49798-default-rtdb.firebaseio.com",
    "projectId": "sw-projects-49798",
    "storageBucket": "sw-projects-49798.firebasestorage.app",
    "messagingSenderId": "812186368395",
    "appId": "1:812186368395:web:be2f7291ce54396209d78e"
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
firestore = firebase.database()
storage = firebase.storage()

# ---------------------
# 세션 상태 초기화 (unchanged)
# ---------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.session_state.id_token = ""
    st.session_state.user_name = ""
    st.session_state.user_gender = "선택 안함"
    st.session_state.user_phone = ""
    st.session_state.profile_image_url = ""

# ---------------------
# 지역명 영어 매핑
# ---------------------
region_map = {
    '전국': 'Nationwide',
    '서울': 'Seoul',
    '부산': 'Busan',
    '대구': 'Daegu',
    '인천': 'Incheon',
    '광주': 'Gwangju',
    '대전': 'Daejeon',
    '울산': 'Ulsan',
    '세종': 'Sejong',
    '경기': 'Gyeonggi',
    '강원': 'Gangwon',
    '충북': 'Chungbuk',
    '충남': 'Chungnam',
    '전북': 'Jeonbuk',
    '전남': 'Jeonnam',
    '경북': 'Gyeongbuk',
    '경남': 'Gyeongnam',
    '제주': 'Jeju'
}

# ---------------------
# 홈 페이지 클래스
# ---------------------
class Home:
    def __init__(self, login_page, register_page, findpw_page):
        st.title("🏠 Home")
        if st.session_state.get("logged_in"):
            st.success(f"{st.session_state.get('user_email')}님 환영합니다.")

        # Dataset Introduction
        st.markdown("""
                ---  
                **Population Trends Dataset**  
                - Source: Custom dataset (`population_trends.csv`)  
                - Description: Contains population, birth, and death data by year and region in South Korea.  
                - Key Columns:  
                  - `year`: Year of the data  
                  - `region`: Region name (e.g., Nationwide, Seoul, Sejong, etc.)  
                  - `population`: Total population  
                  - `births`: Number of births  
                  - `deaths`: Number of deaths  
                """)

# ---------------------
# 로그인 페이지 클래스 (unchanged)
# ---------------------
class Login:
    def __init__(self):
        st.title("🔐 Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.id_token = user['idToken']

                user_info = firestore.child("users").child(email.replace(".", "_")).get().val()
                if user_info:
                    st.session_state.user_name = user_info.get("name", "")
                    st.session_state.user_gender = user_info.get("gender", "선택 안함")
                    st.session_state.user_phone = user_info.get("phone", "")
                    st.session_state.profile_image_url = user_info.get("profile_image_url", "")

                st.success("Login successful!")
                time.sleep(1)
                st.rerun()
            except Exception:
                st.error("Login failed")

# ---------------------
# 회원가입 페이지 클래스 (unchanged)
# ---------------------
class Register:
    def __init__(self, login_page_url):
        st.title("📝 Register")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        name = st.text_input("Name")
        gender = st.selectbox("Gender", ["선택 안함", "Male", "Female"])
        phone = st.text_input("Phone Number")

        if st.button("Register"):
            try:
                auth.create_user_with_email_and_password(email, password)
                firestore.child("users").child(email.replace(".", "_")).set({
                    "email": email,
                    "name": name,
                    "gender": gender,
                    "phone": phone,
                    "role": "user",
                    "profile_image_url": ""
                })
                st.success("Registration successful! Redirecting to login page.")
                time.sleep(1)
                st.switch_page(login_page_url)
            except Exception:
                st.error("Registration failed")

# ---------------------
# 비밀번호 찾기 페이지 클래스 (unchanged)
# ---------------------
class FindPassword:
    def __init__(self):
        st.title("🔎 Find Password")
        email = st.text_input("Email")
        if st.button("Send Password Reset Email"):
            try:
                auth.send_password_reset_email(email)
                st.success("Password reset email sent.")
                time.sleep(1)
                st.rerun()
            except:
                st.error("Email sending failed")

# ---------------------
# 사용자 정보 수정 페이지 클래스 (unchanged)
# ---------------------
class UserInfo:
    def __init__(self):
        st.title("👤 User Info")
        email = st.session_state.get("user_email", "")
        new_email = st.text_input("Email", value=email)
        name = st.text_input("Name", value=st.session_state.get("user_name", ""))
        gender = st.selectbox(
            "Gender",
            ["선택 안함", "Male", "Female"],
            index=["선택 안함", "Male", "Female"].index(st.session_state.get("user_gender", "선택 안함"))
        )
        phone = st.text_input("Phone Number", value=st.session_state.get("user_phone", ""))

        uploaded_file = st.file_uploader("Profile Image Upload", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_path = f"profiles/{email.replace('.', '_')}.jpg"
            storage.child(file_path).put(uploaded_file, st.session_state.id_token)
            image_url = storage.child(file_path).get_url(st.session_state.id_token)
            st.session_state.profile_image_url = image_url
            st.image(image_url, width=150)
        elif st.session_state.get("profile_image_url"):
            st.image(st.session_state.profile_image_url, width=150)

        if st.button("Update"):
            st.session_state.user_email = new_email
            st.session_state.user_name = name
            st.session_state.user_gender = gender
            st.session_state.user_phone = phone

            firestore.child("users").child(new_email.replace(".", "_")).update({
                "email": new_email,
                "name": name,
                "gender": gender,
                "phone": phone,
                "profile_image_url": st.session_state.get("profile_image_url", "")
            })

            st.success("User info updated.")
            time.sleep(1)
            st.rerun()

# ---------------------
# 로그아웃 페이지 클래스 (unchanged)
# ---------------------
class Logout:
    def __init__(self):
        st.session_state.logged_in = False
        st.session_state.user_email = ""
        st.session_state.id_token = ""
        st.session_state.user_name = ""
        st.session_state.user_gender = "선택 안함"
        st.session_state.user_phone = ""
        st.session_state.profile_image_url = ""
        st.success("Logged out.")
        time.sleep(1)
        st.rerun()

# ---------------------
# EDA 페이지 클래스
# ---------------------
class EDA:
    def __init__(self):
        st.title("📊 Population Trends EDA")
        uploaded = st.file_uploader("Upload dataset (population_trends.csv)", type="csv")
        if not uploaded:
            st.info("Please upload population_trends.csv file.")
            return

        # Load and preprocess data
        df = pd.read_csv(uploaded)
        
        # Preprocessing: Replace '-' with 0 for Sejong and convert columns to numeric
        df.loc[df['region'] == '세종', df.columns] = df.loc[df['region'] == '세종', df.columns].replace('-', 0)
        numeric_columns = ['population', 'births', 'deaths']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Map Korean region names to English
        df['region'] = df['region'].map(region_map)

        tabs = st.tabs([
            "Basic Statistics",
            "Yearly Trend",
            "Regional Analysis",
            "Change Analysis",
            "Visualization"
        ])

        # Tab 1: Basic Statistics
        with tabs[0]:
            st.header("Basic Statistics")
            st.subheader("Dataset Structure")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

            st.subheader("Summary Statistics")
            st.dataframe(df.describe())

        # Tab 2: Yearly Trend
        with tabs[1]:
            st.header("Yearly Population Trend")
            # Filter for Nationwide data
            df_nation = df[df['region'] == 'Nationwide'].copy()
            
            # Predict population for 2035 using linear regression
            X = df_nation[['year']].values
            y = df_nation['population'].values
            model = LinearRegression()
            model.fit(X, y)
            future_year = np.array([[2035]])
            predicted_population = model.predict(future_year)[0]
            
            # Create DataFrame for prediction
            pred_df = pd.DataFrame({
                'year': [2035],
                'region': ['Nationwide'],
                'population': [int(predicted_population)],
                'births': [df_nation['births'].iloc[-3:].mean()],  # Average of last 3 years
                'deaths': [df_nation['deaths'].iloc[-3:].mean()]   # Average of last 3 years
            })
            df_plot = pd.concat([df_nation, pred_df], ignore_index=True)

            # Plot population trend
            fig, ax = plt.subplots()
            sns.lineplot(x='year', y='population', data=df_plot, marker='o', ax=ax)
            ax.scatter(2035, predicted_population, color='red', label='Predicted (2035)')
            ax.set_title("Population Trend")
            ax.set_xlabel("Year")
            ax.set_ylabel("Population")
            ax.legend()
            st.pyplot(fig)

            st.markdown("""
            > **Interpretation**:  
            > The graph shows the population trend for Nationwide data, with a prediction for 2035 based on a linear regression model using historical data. The red dot indicates the predicted population, calculated using the trend from past years and averaged birth/death rates from the last three years.
            """)

        # Tab 3: Regional Analysis
        with tabs[2]:
            st.header("Regional Population Change")
            # Filter out Nationwide and get last 5 years
            df_regional = df[df['region'] != 'Nationwide'].copy()
            recent_years = df_regional['year'].nlargest(5).unique()
            df_recent = df_regional[df_regional['year'].isin(recent_years)]
            
            # Calculate population change
            df_pivot = df_recent.pivot(index='region', columns='year', values='population')
            df_pivot['change'] = df_pivot[recent_years[-1]] - df_pivot[recent_years[0]]
            df_pivot['change_rate'] = (df_pivot['change'] / df_pivot[recent_years[0]] * 100).round(2)
            
            # Convert change to thousands for plotting
            df_pivot['change_thousands'] = df_pivot['change'] / 1000
            
            # Plot population change
            st.subheader("Population Change (Last 5 Years)")
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            sns.barplot(x='change_thousands', y='region', data=df_pivot.sort_values('change', ascending=False), ax=ax1, palette='coolwarm')
            for i, v in enumerate(df_pivot.sort_values('change', ascending=False)['change_thousands']):
                ax1.text(v, i, f'{v:.1f}', va='center')
            ax1.set_title("Population Change by Region")
            ax1.set_xlabel("Change (Thousands)")
            ax1.set_ylabel("Region")
            st.pyplot(fig1)

            # Plot population change rate
            st.subheader("Population Change Rate (Last 5 Years)")
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            sns.barplot(x='change_rate', y='region', data=df_pivot.sort_values('change_rate', ascending=False), ax=ax2, palette='coolwarm')
            for i, v in enumerate(df_pivot.sort_values('change_rate', ascending=False)['change_rate']):
                ax2.text(v, i, f'{v:.1f}%', va='center')
            ax2.set_title("Population Change Rate by Region")
            ax2.set_xlabel("Change Rate (%)")
            ax2.set_ylabel("Region")
            st.pyplot(fig2)

            st.markdown("""
            > **Interpretation**:  
            > The first graph shows the absolute population change (in thousands) over the last 5 years, sorted by magnitude. Positive values indicate population growth, while negative values indicate decline. The second graph shows the percentage change relative to the population 5 years ago, highlighting regions with the highest growth or decline rates. Regions like Sejong may show significant growth due to urban development, while others may show declines due to aging or migration trends.
            """)

        # Tab 4: Change Analysis
        with tabs[3]:
            st.header("Population Change Rankings")
            # Calculate year-over-year population difference
            df_regional = df[df['region'] != 'Nationwide'].copy()
            df_regional['pop_diff'] = df_regional.groupby('region')['population'].diff().fillna(0)
            df_regional['pop_diff_thousands'] = df_regional['pop_diff'] / 1000
            
            # Get top 100 cases by absolute difference
            top_changes = df_regional[['year', 'region', 'pop_diff_thousands']].copy()
            top_changes = top_changes.sort_values(by='pop_diff_thousands', key=abs, ascending=False).head(100)
            
            # Style the table with color gradient
            def color_gradient(val):
                if val > 0:
                    color = f'background-color: rgba(0, 0, 255, {min(val/10, 1)})'  # Blue for increase
                elif val < 0:
                    color = f'background-color: rgba(255, 0, 0, {min(abs(val)/10, 1)})'  # Red for decrease
                else:
                    color = 'background-color: white'
                return color
            
            styled_df = top_changes.style.format({
                'pop_diff_thousands': '{:,.1f}'
            }).applymap(color_gradient, subset=['pop_diff_thousands'])
            
            st.subheader("Top 100 Population Changes")
            st.dataframe(styled_df)

            st.markdown("""
            > **Interpretation**:  
            > The table lists the top 100 year-over-year population changes across regions (excluding Nationwide). Positive values (blue) indicate population increases, while negative values (red) indicate decreases. The color intensity reflects the magnitude of the change, with darker colors representing larger changes. Numbers are displayed in thousands with commas for readability.
            """)

        # Tab 5: Visualization
        with tabs[4]:
            st.header("Population Heatmap")
            # Create pivot table for heatmap
            pivot_table = df.pivot(index='region', columns='year', values='population')
            
            # Plot stacked area chart
            fig, ax = plt.subplots(figsize=(12, 8))
            pivot_table.T.plot(kind='area', stacked=True, ax=ax, cmap='tab20')
            ax.set_title("Population by Region and Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Population")
            ax.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
            > **Interpretation**:  
            > The stacked area chart visualizes the population distribution across regions over time. Each region is represented by a distinct color, with the area size proportional to the population. This chart highlights how the population of each region contributes to the total over the years, with Nationwide typically dominating due to its aggregate nature.
            """)

# ---------------------
# 페이지 객체 생성 (unchanged)
# ---------------------
Page_Login    = st.Page(Login,    title="Login",    icon="🔐", url_path="login")
Page_Register = st.Page(lambda: Register(Page_Login.url_path), title="Register", icon="📝", url_path="register")
Page_FindPW   = st.Page(FindPassword, title="Find PW", icon="🔎", url_path="find-password")
Page_Home     = st.Page(lambda: Home(Page_Login, Page_Register, Page_FindPW), title="Home", icon="🏠", url_path="home", default=True)
Page_User     = st.Page(UserInfo, title="My Info", icon="👤", url_path="user-info")
Page_Logout   = st.Page(Logout,   title="Logout",  icon="🔓", url_path="logout")
Page_EDA      = st.Page(EDA,      title="EDA",     icon="📊", url_path="eda")

# ---------------------
# 네비게이션 실행 (unchanged)
# ---------------------
if st.session_state.logged_in:
    pages = [Page_Home, Page_User, Page_Logout, Page_EDA]
else:
    pages = [Page_Home, Page_Login, Page_Register, Page_FindPW]

selected_page = st.navigation(pages)
selected_page.run()
