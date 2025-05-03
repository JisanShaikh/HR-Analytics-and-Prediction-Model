import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import calendar
import requests
from fuzzywuzzy import fuzz

# ========== MUST BE FIRST ==========
st.set_page_config(
    page_title="HR Management Dashboard",
    page_icon="images/company_logo.webp",
    layout="wide")

# ========== Configuration ==========
HF_TOKEN = "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"  
MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# ========== Data Loading ==========
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

@st.cache_data
def load_resources():
    resources = {}
    try:
        df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        resources['df'] = df
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        resources['df'] = None
    
    try:
        with open('attrition_model_simplified.pkl', 'rb') as f:
            resources['model'] = pickle.load(f)
    except Exception as e:
        st.warning(f"Model loading error: {str(e)}")
        resources['model'] = None
    
    return resources

# Load both datasets
df = load_data()
resources = load_resources()
chatbot_df, model = resources['df'], resources['model']

def get_data_answer(question):
    """Generate answers directly from dataset"""
    if chatbot_df is None:
        return None
    
    question = question.lower()
    
    if any(x in question for x in ["total employees", "how many employees"]):
        return f"Our organization currently has {len(chatbot_df)} employees."
    elif any(x in question for x in ["attrition rate", "turnover rate"]):
        rate = chatbot_df['Attrition'].mean() * 100
        return f"The current attrition rate is {rate:.1f}%."
    elif any(x in question for x in ["highest attrition", "most turnover"]):
        dept_stats = chatbot_df.groupby('Department')['Attrition'].mean().sort_values(ascending=False)
        return "\n".join([f"- {dept}: {rate*100:.1f}%" for dept, rate in dept_stats.items()])
    elif any(x in question for x in ["job satisfaction", "employee satisfaction"]):
        avg = chatbot_df['JobSatisfaction'].mean()
        return f"Average job satisfaction score: {avg:.1f}/5"
    return None
def get_ai_response(question):
    """Get direct answers for metrics, proper responses for general HR questions"""
    question_lower = question.lower().strip()
    
    # 1. Handle greetings
    if any(greet in question_lower for greet in ["hi", "hello", "hey"]):
        return "Hello, how can I help?"
    
    # 2. Direct metric answers (no explanations)
    if chatbot_df is not None:
        if "retention rate" in question_lower:
            return f"{100 - chatbot_df['Attrition'].mean()*100:.1f}%"
        elif any(phrase in question_lower for phrase in ["employees", "company size"]):
            return f"{len(chatbot_df)}"
        elif "attrition" in question_lower:
            return f"{chatbot_df['Attrition'].mean()*100:.1f}%"
    
    # 3. Process general HR questions
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Different prompts for different question types
    if any(phrase in question_lower for phrase in ["email", "template", "write"]):
        prompt = f"""Create a professional HR email about: {question.replace('email', '').replace('template', '')}
        Output ONLY the email body without subject or greetings:"""
    else:
        prompt = f"{question}"
    
   
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL}",
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 100}}
        )
        answer = response.json()[0]['generated_text'].split("Answer:")[-1].strip()
        return answer if answer else "I'd be happy to help with that HR question."
    except Exception as e:
        return "I'm unable to access that information right now. Please try again later."

# ========== Custom CSS ==========
st.markdown("""
<style>
    :root {
        --primary: #2c3e50;
        --secondary: #3498db;
        --accent: #e74c3c;
        --light-bg: #f8f9fa;
        --dark-bg: #2c3e50;
        --text: #333333;
        --text-light: #7f8c8d;
    }
    
    body {
        color: var(--text);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .header-style {
        font-size: 32px;
        font-weight: bold;
        color: white(--primary);
        margin-bottom: 20px;
        border-bottom: 2px solid var(--secondary);
        padding-bottom: 10px;
    }
    
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 4px solid var(--secondary);
    }
    
    .metric-card {
        background-color: var(--light-bg);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .nav-button {
        width: 100%;
        padding: 12px;
        margin: 5px 0;
        text-align: left;
        border: none;
        background-color: var(--light-bg);
        color: var(--primary);
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.3s;
        font-weight: 500;
    }
    
    .nav-button:hover {
        background-color: var(--secondary);
        color: white;
    }
    
    .nav-button.active {
        background-color: var(--primary);
        color: white;
    }
    
    .input-section {
        background-color: var(--light-bg);
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    
    .result-section {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    .stButton>button {
        width: 100%;
        padding: 10px;
        font-weight: bold;
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 6px;
        transition: background-color 0.3s;
    }
    
    .stButton>button:hover {
        background-color: var(--secondary);
    }
    
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        background-color: #ffffff;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    
    .chat-message {
        padding: 12px 16px;
        border-radius: 18px;
        margin-bottom: 12px;
        max-width: 80%;
        line-height: 1.5;
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    
    .assistant-message {
        background-color: #f1f1f1;
        margin-right: auto;
        border-bottom-left-radius: 5px;
    }
    
    .stTabs [role="tablist"] {
        gap: 10px;
    }
    
    .stTabs [role="tab"] {
        padding: 10px 20px;
        border-radius: 6px 6px 0 0;
        background-color: var(--light-bg);
        color: var(--text);
        transition: all 0.3s;
    }
    
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: var(--primary);
        color: white;
    }
    
    .stDataFrame {
        border-radius: 8px;
    }
    
    footer {
        text-align: center;
        padding: 15px;
        margin-top: 30px;
        color: var(--text-light);
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ========== Sidebar Navigation ==========

menu_options = ["Home", "Data Overview", "Attrition Analysis", "Promotion Eligibility", 
               "Skill Gap Analysis", "Attrition Prediction", "Attrition Prediction by EmployeeID",
               "Workforce Management", "HR Chatbot"]

# Initialize session state for selected option
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = "Home"

# Create navigation buttons
for option in menu_options:
    if st.sidebar.button(option, key=f"nav_{option}"):
        st.session_state.selected_option = option

# Add some spacing
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

# ========== Page Content ==========
if st.session_state.selected_option == "Home":
    st.markdown('<div class="header-style">HR Analytics Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:16px;color:#666;">Total Employees</div>
            <div style="font-size:24px;font-weight:bold;color:var(--primary);">{}</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    with col2:
        attrition_rate = df['Attrition'].value_counts()['Yes'] / len(df) * 100
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:16px;color:#666;">Attrition Rate</div>
            <div style="font-size:24px;font-weight:bold;color:var(--accent);">{:.2f}%</div>
        </div>
        """.format(attrition_rate), unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:16px;color:#666;">Avg Monthly Income</div>
            <div style="font-size:24px;font-weight:bold;color:var(--primary);">${:,.2f}</div>
        </div>
        """.format(df['MonthlyIncome'].mean()), unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:16px;color:#666;">Avg Job Satisfaction</div>
            <div style="font-size:24px;font-weight:bold;color:var(--primary);">{:.2f}/4</div>
        </div>
        """.format(df['JobSatisfaction'].mean()), unsafe_allow_html=True)

    st.markdown("### Employee Distribution")
    tab1, tab2 = st.tabs(["By Department", "By Job Role"])
    
    with tab1:
        fig = px.pie(df, names='Department', title='Employee Distribution by Department',
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True) 
    
    with tab2:
        fig = px.bar(df['JobRole'].value_counts(), title='Employee Distribution by Job Role',
                    color_discrete_sequence=[px.colors.qualitative.Pastel[0]])
        st.plotly_chart(fig, use_container_width=True)  

elif st.session_state.selected_option == "Data Overview":
    st.markdown('<div class="header-style">Data Overview</div>', unsafe_allow_html=True)
    
    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(df.head(), use_container_width=True) 
    
    with st.expander("Data Description"):
        st.write(df.describe())
    
    with st.expander("Data Types"):
        st.write(df.dtypes)

elif st.session_state.selected_option == "Attrition Analysis":
    st.markdown('<div class="header-style">Attrition Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Attrition Rate by Department")
        fig = px.bar(pd.crosstab(df['Department'], df['Attrition']), 
                    barmode='group', text_auto=True,
                    color_discrete_sequence=[px.colors.qualitative.Pastel[1], px.colors.qualitative.Pastel[0]])
        st.plotly_chart(fig, use_container_width=True)  
        
    with col2:
        st.markdown("### Attrition by Job Role")
        fig = px.bar(pd.crosstab(df['JobRole'], df['Attrition']), 
                    barmode='group', text_auto=True,
                    color_discrete_sequence=[px.colors.qualitative.Pastel[1], px.colors.qualitative.Pastel[0]])
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True) 
    
    st.markdown("### Job Satisfaction vs. Monthly Income")
    fig = px.scatter(df, x='MonthlyIncome', y='JobSatisfaction', 
                    color='Attrition', hover_data=['JobRole'],
                    color_discrete_sequence=[px.colors.qualitative.Pastel[1], px.colors.qualitative.Pastel[0]])
    st.plotly_chart(fig, use_container_width=True)

elif st.session_state.selected_option == "Promotion Eligibility":
    st.markdown('<div class="header-style">Promotion Eligibility</div>', unsafe_allow_html=True)

    df['PromotionEligibility'] = np.where(
        (df['JobSatisfaction'] >= 3) & 
        (df['PerformanceRating'] >= 3) & 
        (df['YearsAtCompany'] >= 3), "Eligible", "Not Eligible")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Eligibility Overview")
        fig = px.pie(df, names='PromotionEligibility', 
                    title='Promotion Eligibility Distribution',
                    color_discrete_sequence=[px.colors.qualitative.Pastel[1], px.colors.qualitative.Pastel[0]])
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### Eligible Employees by Department")
        fig = px.bar(df[df['PromotionEligibility'] == "Eligible"]['Department'].value_counts(),
                    color_discrete_sequence=[px.colors.qualitative.Pastel[0]])
        st.plotly_chart(fig, use_container_width=True)
      
    st.markdown("### Eligible Employees List")
    st.dataframe(df[df['PromotionEligibility'] == "Eligible"][['EmployeeNumber', 'Department', 'JobRole', 'YearsAtCompany']], 
                use_container_width=True)
  
elif st.session_state.selected_option == "Skill Gap Analysis":
    st.markdown('<div class="header-style">Skill Gap Analysis</div>', unsafe_allow_html=True)
    st.write("Identify skill gaps and recommend training programs across the organization.")

    # Define required skills and training programs
    required_skills = {
        "Research Scientist": ["Data Analysis", "Machine Learning", "Statistics"],
        "Sales Executive": ["Communication", "Negotiation", "Customer Relationship Management"],
        "Laboratory Technician": ["Lab Techniques", "Data Recording", "Safety Protocols"],
        "Manufacturing Director": ["Operations Management", "Supply Chain", "Quality Control"],
        "Healthcare Representative": ["Medical Knowledge", "Customer Service", "Sales"],
        "Manager": ["Leadership", "Strategic Planning", "Team Management"],
        "Sales Representative": ["Sales Techniques", "Product Knowledge", "Communication"],
        "Research Director": ["Research Methodology", "Data Analysis", "Project Management"],
        "Human Resources": ["Recruitment", "Employee Relations", "HR Policies"]
    }

    training_programs = {
        "Data Analysis": "https://www.coursera.org/courses?query=data+analysis&skills=Data+Analysis",
        "Machine Learning": "https://www.coursera.org/courses?query=machine+learning&skills=Machine+Learning",
        "Statistics":  "https://www.coursera.org/courses?query=statistics",
        "Communication": "https://www.udemy.com/topic/communication-skills/",
        "Negotiation": "https://www.udemy.com/topic/negotiation/",
        "Customer Relationship Management": "https://www.coursera.org/courses?query=crm",
        "Lab Techniques": "https://www.udemy.com/course/molbio-expert-y-learn/",
        "Safety Protocols": "https://www.coursera.org/courses?query=safety",
        "Operations Management": "https://www.coursera.org/courses?query=operations+management",
        "Supply Chain": "https://www.coursera.org/specializations/supply-chain-management",
        "Quality Control":   "https://www.coursera.org/courses?query=quality+control",
        "Medical Knowledge": "https://www.coursera.org/courses?query=medical+terminology",
        "Customer Service": "https://www.udemy.com/topic/customer-service/",
        "Sales": "https://www.coursera.org/courses?query=sales",
        "Leadership": "https://www.coursera.org/courses?query=leadership",
        "Strategic Planning": "https://www.coursera.org/courses?query=strategic+planning",
        "Team Management":"https://www.coursera.org/courses?query=team+management",
        "Sales Techniques": "https://www.udemy.com/topic/sales-skills/",
        "Product Knowledge": "https://www.coursera.org/courses?query=product+management",
        "Research Methodology": "https://www.coursera.org/courses?query=research+methods",
        "Project Management": "https://www.coursera.org/courses?query=project+management",
        "Recruitment": "https://www.coursera.org/courses?query=recruitment",
        "Employee Relations": "https://www.coursera.org/courses?query=employee+relations",
        "HR Policies": "https://www.coursera.org/courses?query=hr+policies"
    }

    def recommend_training(job_role, performance_rating, training_times_last_year, job_satisfaction):
        if performance_rating < 3 or training_times_last_year < 2 or job_satisfaction < 3:
            return required_skills.get(job_role, [])
        return []

    # Add filters for department and job role
    selected_department = st.selectbox("Select Department", df['Department'].unique())
    selected_job_role = st.selectbox("Select Job Role", df[df['Department'] == selected_department]['JobRole'].unique())

    filtered_df = df[(df['Department'] == selected_department) & (df['JobRole'] == selected_job_role)]

    skill_gaps = filtered_df.apply(
        lambda row: recommend_training(row['JobRole'], row['PerformanceRating'], row['TrainingTimesLastYear'], row['JobSatisfaction']), 
        axis=1)

    all_skill_gaps = [skill for sublist in skill_gaps for skill in sublist]

    st.subheader(f"Skill Gaps in {selected_department} - {selected_job_role}")
    if all_skill_gaps:
        top_skill_gaps = pd.Series(all_skill_gaps).value_counts().head(5)
        st.write("**Top 5 Skill Gaps:**")
        st.write(top_skill_gaps)

        st.subheader("Recommended Training Programs")
        for skill in top_skill_gaps.index:
            if skill in training_programs:
                st.write(f"- **{skill}**: [Course Link]({training_programs[skill]})")
            else:
                st.write(f"- **{skill}**: No training link available.")
    else:
        st.write("No significant skill gaps found.")

elif st.session_state.selected_option == "Attrition Prediction":
    st.markdown('<div class="header-style">Employee Attrition Risk Dashboard</div>', unsafe_allow_html=True)
    
    @st.cache_resource
    def load_model():
        try:
            with open('simple_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            st.stop()

    model = load_model()

    SELECTED_FEATURES = ['MonthlyIncome', 'Age', 'JobSatisfaction', 
                        'OverTime', 'YearsAtCompany']

    with st.container():
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            with st.form("employee_form"):
                st.markdown("### Employee Details")

                st.markdown("**Personal Information**")
                age = st.text_input("Age (Years)", value="30", 
                                  help="Enter employee's age in years")
                
                monthly_income = st.text_input("Monthly Salary (₹)", value="65000",
                                             help="Enter gross monthly salary in INR")

                st.markdown("**Employment Information**")
                years_at_company = st.text_input("Tenure (Years)", value="3",
                                               help="Years with the company (e.g. 2.5)")
                
                job_satisfaction = st.text_input("Job Satisfaction (1-4)", value="3",
                                               help="1=Very Low, 2=Low, 3=High, 4=Very High")
                
                overtime = st.radio("Overtime Frequently", ["No", "Yes"], horizontal=True)
                
                last_promotion = st.text_input("Years Since Last Promotion", value="2",
                                             help="Time since last promotion")
                
                submitted = st.form_submit_button("Calculate Attrition Risk", 
                                                use_container_width=True)

        with col2:
            if submitted:
                try:
                    age = float(age)
                    monthly_income = float(monthly_income)
                    years_at_company = float(years_at_company)
                    job_satisfaction = float(job_satisfaction)
                    last_promotion = float(last_promotion)
                    
                    if not (1 <= job_satisfaction <= 4):
                        st.error("Job Satisfaction must be between 1 and 4")
                        st.stop()
                        
                    if any(val < 0 for val in [age, monthly_income, years_at_company, last_promotion]):
                        st.error("Negative values are not allowed")
                        st.stop()
                        
                except ValueError:
                    st.error("Please enter valid numbers in all fields")
                    st.stop()

                input_data = {
                    'MonthlyIncome': monthly_income,
                    'Age': age,
                    'JobSatisfaction': job_satisfaction,
                    'OverTime': 1 if overtime == "Yes" else 0,
                    'YearsAtCompany': years_at_company
                }
                
                try:
                    input_df = pd.DataFrame([input_data])[SELECTED_FEATURES]
                    attrition_prob = model.predict_proba(input_df)[0][1] * 100
                    retention_prob = 100 - attrition_prob

                    with st.container():
                        st.markdown("### Risk Assessment")
 
                        st.metric(label="ATTRITION RISK", 
                                 value=f"{attrition_prob:.1f}%",
                                 delta_color="inverse",
                                 help="Probability the employee will leave within 12 months",
                                 label_visibility="visible")
  
                        st.metric(label="Retention Probability", 
                                 value=f"{retention_prob:.1f}%",
                                 help="Likelihood the employee will stay",
                                 label_visibility="visible")
      
                        if attrition_prob > 40:
                            risk_level = "🔴 Critical Risk"
                            color = "#e74c3c"
                        elif attrition_prob > 25:
                            risk_level = "🟠 High Risk"
                            color = "#f39c12"
                        elif attrition_prob > 15:
                            risk_level = "🟡 Moderate Risk"
                            color = "#f1c40f"
                        else:
                            risk_level = "🟢 Low Risk"
                            color = "#2ecc71"
                        
                        st.markdown(f"""<div style="background-color:{color}20; padding:15px; border-radius:10px; margin:15px 0;"> <h4     style="color:{color}; margin:0;">{risk_level}</h4>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("### Recommended Actions")
                        
                        if attrition_prob > 40:
                            st.error("**Immediate Intervention Required**")
                            st.markdown("""
                            - Schedule emergency 1:1 within **48 hours**
                            - Conduct compensation review immediately
                            - Develop personalized retention package
                            - Assign executive mentor
                            """)
                        elif attrition_prob > 25:
                            st.warning("**Priority Attention Needed**")
                            st.markdown("""
                            - Conduct stay interview within **1 week**
                            - Review career progression path
                            - Consider spot bonus or equity grant
                            - Increase check-in frequency
                            """)
                        elif attrition_prob > 15:
                            st.info("**Monitor Closely**")
                            st.markdown("""
                            - Schedule development discussion
                            - Recognize recent contributions
                            - Assess workload balance
                            - Quarterly retention review
                            """)
                        else:
                            st.success("**Stable Retention**")
                            st.markdown("""
                            - Continue regular engagement
                            - Maintain development opportunities
                            - Annual retention check
                            """)
                            
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

elif st.session_state.selected_option == "Attrition Prediction by EmployeeID":
    st.markdown('<div class="header-style">Employee Attrition Risk Prediction</div>', unsafe_allow_html=True)
    
    # Load the model
    @st.cache_data
    def load_model():
        try:
            with open('attrition_model_simplified.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            st.stop()

    model = load_model()
    
    # Create a single input form
    with st.form("employee_id_form"):
        st.markdown("### Enter Employee ID")
        employee_id = st.text_input(
            "Employee ID",
            help="Enter the employee ID to predict attrition risk",
            placeholder="e.g. 1234"
        )
        
        submitted = st.form_submit_button("Predict Attrition Risk")
    
    if submitted:
        # Check if employee exists
        if employee_id.strip() == "":
            st.warning("Please enter an Employee ID")
            st.stop()
            
        try:
            employee_id = int(employee_id)  # Convert to integer
        except ValueError:
            st.error("Employee ID must be a number")
            st.stop()
            
        if employee_id not in df['EmployeeNumber'].values:
            st.error(f"Employee ID {employee_id} not found in database")
            st.stop()
            
        # Get the employee's data
        employee = df[df['EmployeeNumber'] == employee_id].iloc[0]
        
        # Prepare input data - convert categorical to numerical if needed
        overtime_num = 1 if employee['OverTime'] == 'Yes' else 0
        
        input_data = {
            'MonthlyIncome': employee['MonthlyIncome'],
            'Age': employee['Age'],
            'JobSatisfaction': employee['JobSatisfaction'],
            'OverTime': overtime_num,
            'YearsAtCompany': employee['YearsAtCompany']
        }
        
        # Make prediction
        input_df = pd.DataFrame([input_data])
        attrition_prob = model.predict_proba(input_df)[0][1] * 100
        retention_prob = 100 - attrition_prob
        
        # Display results in the same format as your original
        st.divider()
        st.markdown(f"### Prediction Results for Employee {employee_id}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Employee Details**")
            st.write(f"- **Department:** {employee['Department']}")
            st.write(f"- **Job Role:** {employee['JobRole']}")
            st.write(f"- **Monthly Income:** ${employee['MonthlyIncome']:,.2f}")
            st.write(f"- **Age:** {employee['Age']} years")
            st.write(f"- **Tenure:** {employee['YearsAtCompany']} years")
            
        with col2:
            st.markdown("**Risk Assessment**")
            st.metric(label="ATTRITION RISK", 
                     value=f"{attrition_prob:.1f}%",
                     delta_color="inverse")
            
            # Risk level indicator
            if attrition_prob > 40:
                risk_level = "🔴 Critical Risk"
                color = "#e74c3c"
            elif attrition_prob > 25:
                risk_level = "🟠 High Risk"
                color = "#f39c12"
            elif attrition_prob > 15:
                risk_level = "🟡 Moderate Risk"
                color = "#f1c40f"
            else:
                risk_level = "🟢 Low Risk"
                color = "#2ecc71"
            
            st.markdown(f"""<div style="background-color:{color}20; padding:15px; border-radius:10px; margin-top:10px;">
                            <p style="color:{color}; margin:0; font-weight:bold;">{risk_level}</p>
                            </div>""", 
                        unsafe_allow_html=True)
            
            st.metric(label="RETENTION PROBABILITY", 
                     value=f"{retention_prob:.1f}%")
        
        # Recommended actions (same as your original)
        st.divider()
        st.markdown("**Recommended Retention Actions**")
        
        if attrition_prob > 40:
            st.error("**Immediate Intervention Required**")
            st.markdown("""
            - Schedule emergency 1:1 within **48 hours**
            - Conduct compensation review immediately
            - Develop personalized retention package
            - Assign executive mentor
            """)
        elif attrition_prob > 25:
            st.warning("**Priority Attention Needed**")
            st.markdown("""
            - Conduct stay interview within **1 week**
            - Review career progression path
            - Consider spot bonus or equity grant
            - Increase check-in frequency
            """)
        elif attrition_prob > 15:
            st.info("**Monitor Closely**")
            st.markdown("""
            - Schedule development discussion
            - Recognize recent contributions
            - Assess workload balance
            - Quarterly retention review
            """)
        else:
            st.success("**Stable Retention**")
            st.markdown("""
            - Continue regular engagement
            - Maintain development opportunities
            - Annual retention review
            """)

elif st.session_state.selected_option == "Workforce Management":
    st.markdown('<div class="header-style">Workforce Management</div>', unsafe_allow_html=True)
    
    # Workforce Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_headcount = len(df)
        st.metric("Total Headcount", total_headcount)
    
    with col2:
        avg_tenure = df['YearsAtCompany'].mean()
        st.metric("Avg Company Tenure", f"{avg_tenure:.1f} years")
    
    with col3:
        # Vacancy Rate Input
        vacancy_rate = st.number_input(
            "Vacancy Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=3.0,
            step=0.5,
            help="Enter your target vacancy rate percentage"
        )
        open_positions = int(len(df) * (vacancy_rate/100))
        st.metric("Open Positions", open_positions)
    
    with col4:
        turnover_rate = (df['Attrition'].value_counts().get('Yes', 0) / len(df)) * 100
        st.metric("Annual Turnover Rate", f"{turnover_rate:.1f}%")
    
    # Workforce Planning Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Headcount Planning", "Succession Planning", 
                                     "Workforce Costs", "Team Productivity"])
    
    with tab1:
        st.subheader("Headcount Planning")
        
        col1, col2 = st.columns(2)
        with col1:
            # Departmental Headcount visualization
            dept_counts = df['Department'].value_counts().reset_index()
            dept_counts.columns = ['Department', 'Current']
            fig = px.bar(dept_counts, 
                         x='Department', 
                         y='Current',
                         title="Current Headcount by Department",
                         color_discrete_sequence=[px.colors.qualitative.Pastel[0]])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Interactive Hiring Plan Form
            with st.form("hiring_plan_form"):
                st.markdown("**Set Hiring Targets**")
                
                # Get unique departments
                departments = df['Department'].unique()
                
                # Create input fields for each department
                hiring_targets = {}
                for dept in departments:
                    current = len(df[df['Department'] == dept])
                    hiring_targets[dept] = st.number_input(
                        f"{dept} Target Headcount",
                        min_value=0,
                        value=int(current * 1.1),  # Default to 10% growth
                        step=1,
                        key=f"target_{dept}"
                    )
                
                submitted = st.form_submit_button("Update Hiring Plan")
            
            # Display hiring plan after submission
            st.markdown("**Hiring Plan**")
            hiring_data = []
            for dept in departments:
                current = len(df[df['Department'] == dept])
                planned = hiring_targets[dept]
                hiring_data.append({
                    'Department': dept,
                    'Current': current,
                    'Planned': planned,
                    'To Hire': max(0, planned - current)
                })
            
            hiring_df = pd.DataFrame(hiring_data)
            st.dataframe(hiring_df, hide_index=True)
            
            # Hiring Timeline based on inputs
            st.markdown("**Hiring Timeline**")
            total_to_hire = hiring_df['To Hire'].sum()
            
            if total_to_hire > 0:
                q1_percent = st.slider(
                    "Q1 Hiring %", 
                    min_value=0, 
                    max_value=100, 
                    value=40,
                    help="Percentage of hiring to complete in Q1"
                )
                q2_percent = st.slider(
                    "Q2 Hiring %", 
                    min_value=0, 
                    max_value=100-q1_percent, 
                    value=30
                )
                q3_percent = st.slider(
                    "Q3 Hiring %", 
                    min_value=0, 
                    max_value=100-q1_percent-q2_percent, 
                    value=20
                )
                q4_percent = 100 - q1_percent - q2_percent - q3_percent
                
                timeline_data = {
                    'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
                    'Planned Hires': [
                        int(total_to_hire * q1_percent/100),
                        int(total_to_hire * q2_percent/100),
                        int(total_to_hire * q3_percent/100),
                        total_to_hire - int(total_to_hire * q1_percent/100) - 
                        int(total_to_hire * q2_percent/100) - 
                        int(total_to_hire * q3_percent/100)
                    ]
                }
                
                fig = px.bar(timeline_data, x='Quarter', y='Planned Hires',
                            title="Quarterly Hiring Plan",
                            color_discrete_sequence=[px.colors.qualitative.Pastel[0]])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hiring planned based on current targets")

    with tab2:
        st.subheader("Succession Planning")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**High Potential Employees**")
            # Get top performers
            top_performers = df[df['PerformanceRating'] >= 3].sort_values('PerformanceRating', ascending=False)
            st.dataframe(top_performers[['EmployeeNumber', 'Department', 'JobRole', 'YearsAtCompany', 'PerformanceRating']].head(10),
                        hide_index=True)
        
        with col2:
            st.markdown("**Succession Pipeline**")
            
            # Create succession pipeline from actual data
            succession_data = []
            
            # Get top performers by department (potential successors)
            top_performers = df[df['PerformanceRating'] >= 3].sort_values(['Department', 'PerformanceRating'], ascending=False)
            
            for dept in df['Department'].unique():
                dept_employees = top_performers[top_performers['Department'] == dept]
                
                # Get potential successors (top 3 performers in each department)
                ready_now = dept_employees.head(3)['EmployeeName'].tolist() if 'EmployeeName' in df.columns else \
                           dept_employees.head(3)['EmployeeNumber'].astype(str).tolist()
                
                # Get next tier (next 3 performers)
                future_potential = dept_employees.iloc[3:6]['EmployeeName'].tolist() if 'EmployeeName' in df.columns else \
                                  dept_employees.iloc[3:6]['EmployeeNumber'].astype(str).tolist()
                
                # Get leadership roles for each department
                if dept == 'Sales':
                    role = 'Sales Director'
                elif dept == 'Research & Development':
                    role = 'R&D Manager'
                else:
                    role = f'{dept} Lead'
                
                succession_data.append({
                    'Critical Role': role,
                    'Ready Now': ', '.join(ready_now) if ready_now else 'No candidates',
                    '1-2 Years Away': ', '.join(future_potential) if future_potential else 'No candidates'
                })
            
            st.dataframe(pd.DataFrame(succession_data), hide_index=True)

        st.markdown("**Development Plans**")
        with st.expander("View Development Programs"):
            # Create development programs based on actual needs
            programs_data = []
            
            # Leadership program (for high potential employees)
            hipo_count = len(df[(df['PerformanceRating'] >= 3) & (df['YearsAtCompany'] >= 2)])
            programs_data.append({
                'Program': 'Leadership Academy',
                'Participants': hipo_count,
                'Start Date': '2025-09-01',
                'Eligibility': 'High performers with 2+ years tenure'
            })
            
            # Technical training (for technical roles)
            if 'JobRole' in df.columns:
                tech_roles = ['Research Scientist', 'Laboratory Technician', 'Manufacturing Director']
                tech_count = len(df[df['JobRole'].isin(tech_roles)])
                programs_data.append({
                    'Program': 'Technical Mastery',
                    'Participants': tech_count,
                    'Start Date': '2025-10-15',
                    'Eligibility': 'Technical roles requiring certification'
                })
            
            # Management training (for managers and high potentials)
            if 'JobRole' in df.columns:
                # Fixed the manager count with proper NA handling
                mgmt_count = len(df[df['JobRole'].str.contains('Manager', case=False, na=False)])
                programs_data.append({
                    'Program': 'Management Training',
                    'Participants': mgmt_count + hipo_count//2,  # Managers + half of high potentials
                    'Start Date': '2025-11-01',
                    'Eligibility': 'Current managers + high potentials'
                })
            
            st.dataframe(pd.DataFrame(programs_data), hide_index=True)


    with tab3:
        st.subheader("Workforce Costs")
        
        col1, col2 = st.columns(2)
        with col1:
            # Salary Distribution
            fig = px.box(df, y='MonthlyIncome', x='Department',
                        title="Salary Distribution by Department",
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cost Breakdown
            st.markdown("**Annual Cost Breakdown**")
            cost_data = {
                'Category': ['Salaries', 'Benefits', 'Training', 'Recruiting'],
                'Amount ($M)': [25.4, 5.2, 0.8, 0.6],
                '% of Total': [80, 16, 3, 2]
            }
            st.dataframe(pd.DataFrame(cost_data), hide_index=True)
        
        # Cost Projection
        st.markdown("**3-Year Cost Projection**")
        projection_data = {
            'Year': [2023, 2024, 2025],
            'Base Salary': [25.4, 26.7, 28.0],
            'Benefits': [5.2, 5.5, 5.8],
            'Total': [30.6, 32.2, 33.8]
        }
        fig = px.line(projection_data, x='Year', y=['Base Salary', 'Benefits', 'Total'],
                     title="Workforce Cost Projection ($M)",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Team Productivity")
        
        # Productivity Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Projects/Employee", "4.2")
        with col2:
            st.metric("Avg Training Hours", "32.5")
        with col3:
            st.metric("Overtime %", "18.3%")
        
        # Productivity Trends
        st.markdown("**Monthly Productivity Trends**")
        productivity_data = {
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'Output': [85, 88, 92, 90, 94, 96],
            'Quality': [92, 90, 91, 93, 94, 95]
        }
        fig = px.line(productivity_data, x='Month', y=['Output', 'Quality'],
                     title="Team Performance Metrics",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance vs Satisfaction
        st.markdown("**Performance vs Satisfaction**")
        fig = px.scatter(df, x='JobSatisfaction', y='PerformanceRating',
                        color='Department', hover_data=['JobRole'],
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.selected_option == "HR Chatbot":
    st.markdown('<div class="header-style">HR Analytics Assistant</div>', unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Main chat interface
    question = st.chat_input("Ask about our workforce analytics...")

    if question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = get_data_answer(question) or get_ai_response(question)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    st.caption("Try questions like: 'What's our attrition rate?', 'How satisfied are employees?', or 'Which department has highest turnover?'")

# ========== Footer ==========
st.markdown("""
<footer>
    <hr>
    <p><strong>HR Analytics Dashboard</strong> | Model | 
    <a href="mailto:support@hr-analytics.com" style="color: var(--secondary);">Report Issues</a> | © 2025 Jisan Shaikh</p>
</footer>
""", unsafe_allow_html=True)