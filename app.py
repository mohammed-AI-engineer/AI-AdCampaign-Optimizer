import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# 1. UI Configuration (Cyberpunk Style)
# ==========================================
st.set_page_config(layout="wide", page_title="AI AdTech Commander", page_icon="🚀")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #00FF88, #00B8D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .uncopyable {
        user-select: none;
        background: rgba(0, 255, 136, 0.05);
        border-left: 3px solid #00FF88;
        padding: 10px;
        margin-bottom: 5px;
        font-family: monospace;
        color: #E6E6E6;
        cursor: not-allowed;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00C853 0%, #009624 100%);
        border: none;
        color: white;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(0, 200, 83, 0.6);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Asset Loader (Relative Paths)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    # File names should match your exported model files
    model_path = os.path.join(BASE_DIR, "user_demographics_model.pkl")
    encoder_path = os.path.join(BASE_DIR, "label_encoder_dict.pkl")
    
    try:
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            clf = joblib.load(model_path)
            le = joblib.load(encoder_path)
            return clf, le
        else:
            return None, None
    except Exception as e:
        st.error(f"Loading Error: {e}")
        return None, None

clf, le_dict = load_assets()

if clf is None:
    st.error("🚨 **CRITICAL ERROR:** Model or Encoder files not found in the project directory!")
    st.info("💡 Make sure 'xgb_model_custom.pkl' and 'le_dict_xgb_custom.pkl' are in the same folder as this script.")
    st.stop()

def safe_encode(encoder, value):
    try:
        return encoder.transform([str(value).strip()])[0]
    except:
        return 0 

# ==========================================
# 3. Sidebar: Manual Simulator
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80)
    st.title("🕹️ Manual Simulator")
    
    with st.form("manual_form"):
        age = st.slider("Customer Age", 18, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        city = st.selectbox("City", ["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya", "Konya"])
        device = st.selectbox("Device", ["Mobile", "Desktop", "Tablet"])
        payment = st.selectbox("Payment", ["Credit Card", "Digital Wallet", "Cash on Delivery", "Bank Transfer"])
        test_group = st.selectbox("Ad Test Group", ["Test A", "Test B", "Test C"])
        
        predict_btn = st.form_submit_button("🔮 Predict Outcome")
    
    if predict_btn:
        input_data = pd.DataFrame([[age, gender, city, device, payment, test_group]], 
                                  columns=['Age','Gender','City','Device_Type','Payment_Method','Test_Group'])
        for col in input_data.columns:
            if col != 'Age':
                input_data[col] = safe_encode(le_dict[col], input_data[col][0])
        
        prob = clf.predict_proba(input_data)[0][1] * 100 
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Buying Probability"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00FF88" if prob > 50 else "#FF3D00"}}
        ))
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig, use_container_width=True)
        
        if prob > 50:
            st.success("✅ Target User")
        else:
            st.error("🛑 Avoid User")

# ==========================================
# 4. Main Dashboard: Bulk Processing
# ==========================================
st.title("🚀 AI Ad-Campaign Optimizer")
st.markdown("##### Process leads and discover high-conversion opportunities.")

tab_input, tab_charts = st.tabs(["📥 Bulk Processor", "📊 Smart Insights"])

with tab_input:
    example_format = "33,Female,Izmir,Tablet,Credit Card,user@example.org,Test A"
    raw_data = st.text_area("Paste Lead Data (CSV Format)", height=150, placeholder=f"Example: {example_format}")
    
    if st.button("⚡ Run AI Analysis"):
        if raw_data:
            lines = raw_data.strip().split('\n')
            valid_rows = []
            rejected_count = 0

            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                
                # Logic: We process the first 7 columns based on model training
                if len(parts) >= 7: 
                    age_str, gender, city, device, pay, email, test = parts[:7]
                    
                    is_valid = True
                    if not age_str.isdigit(): is_valid = False
                    elif int(age_str) < 18 or int(age_str) > 100: is_valid = False
                    if test == "" or email == "": is_valid = False
                    
                    if is_valid:
                        valid_rows.append([int(age_str), gender, city, device, pay, email, test])
                    else:
                        rejected_count += 1
                else:
                    rejected_count += 1
            
            if valid_rows:
                df = pd.DataFrame(valid_rows, columns=['Age','Gender','City','Device_Type','Payment_Method','Email','Test_Group'])
                
                X_input = df.drop(columns=['Email']).copy()
                for col in ['Gender','City','Device_Type','Payment_Method','Test_Group']:
                    X_input[col] = X_input[col].apply(lambda x: safe_encode(le_dict[col], x))
                
                probs = clf.predict_proba(X_input)[:, 1]
                df['Score'] = (probs * 100).astype(int)
                
                winners = df[df['Score'] >= 50].sort_values(by='Score', ascending=False)
                
                st.session_state['analyzed_df'] = df 
                st.session_state['analyzed_winners'] = winners
                st.session_state['has_run'] = True
                
                m1, m2, m3 = st.columns(3)
                m1.metric("📥 Total Leads", len(valid_rows))
                m2.metric("💎 AI Winners", len(winners))
                m3.metric("🚫 Rejected", rejected_count)
                
                st.divider()
                st.markdown("### 🔒 VIP Selection (High Probability)")
                
                if not winners.empty:
                    for _, row in winners.iterrows():
                        st.markdown(f"""
                        <div class="uncopyable">
                            ✉️ {row['Email']} | ⭐ {row['Score']}% | 📍 {row['City']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    csv = winners.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download VIP Leads CSV", csv, "ai_optimized_leads.csv", "text/csv")
                else:
                    st.info("No high-potential winners detected in this batch.")
            else:
                st.error("❌ Data format error. Required: Age, Gender, City, Device, Payment, Email, Test Group")
        else:
            st.warning("Please paste lead data to begin analysis.")

# ==========================================
# 5. Tab 2: Visualization & Market Intel
# ==========================================
with tab_charts:
    if st.session_state.get('has_run'):
        df_viz = st.session_state['analyzed_df']
        winners_viz = st.session_state['analyzed_winners']
        
        st.header("📈 Market Intelligence")
        
        
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🏆 Ad Test Performance")
            test_scores = df_viz.groupby('Test_Group')['Score'].mean().reset_index()
            fig_test = px.bar(test_scores, x='Test_Group', y='Score', color='Test_Group', 
                            color_discrete_sequence=['#00C853', '#00B8D4', '#FF3D00'])
            st.plotly_chart(fig_test, use_container_width=True)
            
        with c2:
            st.subheader("🌍 Geographical Distribution")
            if not winners_viz.empty:
                city_counts = winners_viz['City'].value_counts().reset_index()
                city_counts.columns = ['City', 'Count']
                fig_city = px.bar(city_counts, x='City', y='Count', text='Count', color='Count', color_continuous_scale='Greens')
                st.plotly_chart(fig_city, use_container_width=True)
            else:
                st.info("No geographical data for winners yet.")
    else:
        st.info("👈 Analysis must be run in the Bulk Processor tab first.")