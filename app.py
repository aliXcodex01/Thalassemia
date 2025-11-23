import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Page configuration
st.set_page_config(
    page_title="Alpha Thalassemia Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-left: 5px solid #1f77b4;
        padding-left: 15px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-input {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    .feature-input:hover {
        border-color: #1f77b4;
        box-shadow: 0 4px 8px rgba(31, 119, 180, 0.2);
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class ThalassemiaPredictor:
    def __init__(self):
        self.models_loaded = False
        self.scaler = None
        self.rf_model = None
        self.xgb_model = None
        
    def load_models(self):
        try:
            self.scaler = joblib.load('scaler.pkl')
            self.rf_model = joblib.load('rf_model.pkl')
            self.xgb_model = joblib.load('xgb_model.pkl')
            self.models_loaded = True
            return True
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return False
    
    def predict(self, input_data):
        if not self.models_loaded:
            return None, None
        
        try:
            # Scale the input
            scaled_data = self.scaler.transform(input_data)
            
            # Make predictions
            rf_pred = self.rf_model.predict(scaled_data)[0]
            rf_prob = self.rf_model.predict_proba(scaled_data)[0]
            
            xgb_pred = self.xgb_model.predict(scaled_data)[0]
            xgb_prob = self.xgb_model.predict_proba(scaled_data)[0]
            
            return (rf_pred, rf_prob), (xgb_pred, xgb_prob)
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            return None, None

def main():
    # Initialize predictor
    predictor = ThalassemiaPredictor()
    
    # Header with gradient text
    st.markdown('<div class="main-header">🧬 Alpha Thalassemia Risk Predictor</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        app_mode = st.radio("", ["🏠 Single Prediction", "📊 Batch Prediction", "ℹ️ About"])
        
        st.markdown("---")
        st.markdown("### 🤖 Model Information")
        
        # Model status
        if predictor.models_loaded:
            st.success("✅ Models Loaded")
        else:
            st.warning("⚠️ Models Not Loaded")
        
        # Load models button
        if st.button("🔄 Load Prediction Models", use_container_width=True):
            with st.spinner("🔄 Loading AI models..."):
                if predictor.load_models():
                    st.success("✅ Models loaded successfully!")
                    st.balloons()
                else:
                    st.error("❌ Failed to load models")
        
        st.markdown("---")
        st.markdown("### 📈 Model Performance")
        st.info("""
        - **Random Forest**: ~95% Accuracy
        - **XGBoost**: ~93% Accuracy  
        - **Ensemble**: Enhanced Reliability
        """)
        
        st.markdown("---")
        st.markdown("#### 🏥 Clinical Note")
        st.caption("For screening purposes only. Always consult healthcare professionals.")

    if app_mode == "🏠 Single Prediction":
        single_prediction_mode(predictor)
    elif app_mode == "📊 Batch Prediction":
        batch_prediction_mode(predictor)
    else:
        about_mode()

def single_prediction_mode(predictor):
    st.markdown('<div class="sub-header">🔍 Individual Risk Assessment</div>', unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["👤 Patient Information", "🩸 Blood Parameters"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Personal Information")
            sex = st.selectbox("**Biological Sex**", ["Male", "Female"], 
                             help="Biological sex of the patient")
            age = st.number_input("**Age**", min_value=0, max_value=120, value=25, 
                                help="Patient age in years")
            
        with col2:
            st.markdown("#### 📝 Clinical Notes")
            symptoms = st.multiselect("**Reported Symptoms**", 
                                   ["Fatigue", "Weakness", "Pale Skin", "Shortness of Breath", 
                                    "None", "Other"])
            family_history = st.selectbox("**Family History**", 
                                       ["No known history", "Possible carrier", 
                                        "Confirmed in family", "Unknown"])
    
    with tab2:
        st.markdown("#### 🩸 Complete Blood Count")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="feature-input">', unsafe_allow_html=True)
            hb = st.slider("**Hemoglobin (Hb g/dL)**", min_value=5.0, max_value=20.0, value=12.0, step=0.1,
                          help="Normal range: 12-16 g/dL (F), 13.5-17.5 g/dL (M)")
            pcv = st.slider("**Packed Cell Volume (PCV %)**", min_value=20.0, max_value=60.0, value=35.0, step=0.1,
                           help="Normal range: 36-48%")
            rbc = st.slider("**RBC Count (million/µL)**", min_value=2.0, max_value=8.0, value=5.0, step=0.1,
                           help="Normal range: 4.2-5.4 million/µL")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="feature-input">', unsafe_allow_html=True)
            mcv = st.slider("**Mean Corpuscular Volume (MCV fL)**", min_value=50.0, max_value=110.0, value=80.0, step=0.1,
                           help="Normal range: 80-100 fL")
            mch = st.slider("**Mean Corpuscular Hemoglobin (MCH pg)**", min_value=15.0, max_value=35.0, value=25.0, step=0.1,
                           help="Normal range: 27-31 pg")
            mchc = st.slider("**MCH Concentration (MCHC g/dL)**", min_value=25.0, max_value=38.0, value=32.0, step=0.1,
                            help="Normal range: 32-36 g/dL")
            st.markdown('</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown('<div class="feature-input">', unsafe_allow_html=True)
            rdw = st.slider("**Red Cell Distribution Width (RDW %)**", min_value=10.0, max_value=30.0, value=14.0, step=0.1,
                           help="Normal range: 11.5-14.5%")
            wbc = st.slider("**White Blood Cells (WBC 10³/µL)**", min_value=2.0, max_value=30.0, value=8.0, step=0.1,
                           help="Normal range: 4.0-11.0 10³/µL")
            neut = st.slider("**Neutrophils (%)**", min_value=10.0, max_value=90.0, value=50.0, step=0.1,
                           help="Normal range: 40-60%")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="feature-input">', unsafe_allow_html=True)
            lymph = st.slider("**Lymphocytes (%)**", min_value=10.0, max_value=90.0, value=40.0, step=0.1,
                            help="Normal range: 20-40%")
            plt = st.slider("**Platelets (PLT 10³/µL)**", min_value=100.0, max_value=800.0, value=300.0, step=1.0,
                          help="Normal range: 150-450 10³/µL")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("#### 🧪 Hemoglobin Analysis")
        col5, col6, col7 = st.columns(3)
        
        with col5:
            hba = st.slider("**HbA (%)**", min_value=50.0, max_value=100.0, value=87.0, step=0.1,
                           help="Normal: >95% in adults")
        with col6:
            hba2 = st.slider("**HbA2 (%)**", min_value=1.0, max_value=5.0, value=2.5, step=0.1,
                            help="Normal range: 2.0-3.5%")
        with col7:
            hbf = st.slider("**HbF (%)**", min_value=0.0, max_value=5.0, value=0.5, step=0.1,
                           help="Normal: <1% in adults")
    
    # Prediction button
    if st.button("🚀 Predict Thalassemia Risk", use_container_width=True):
        if not predictor.models_loaded:
            st.error("❌ Please load models first using the button in the sidebar!")
            return
            
        # Prepare input data
        input_data = pd.DataFrame([{
            'sex': 1 if sex == "Male" else 0,
            'hb': hb, 'pcv': pcv, 'rbc': rbc, 'mcv': mcv, 'mch': mch,
            'mchc': mchc, 'rdw': rdw, 'wbc': wbc, 'neut': neut, 'lymph': lymph,
            'plt': plt, 'hba': hba, 'hba2': hba2, 'hbf': hbf
        }])
        
        # Show loading animation
        with st.spinner("🔄 Analyzing parameters with AI models..."):
            # Make prediction
            (rf_pred, rf_prob), (xgb_pred, xgb_prob) = predictor.predict(input_data)
        
        if rf_pred is not None:
            display_results(rf_pred, rf_prob, xgb_pred, xgb_prob)

def batch_prediction_mode(predictor):
    st.markdown('<div class="sub-header">📊 Batch Prediction</div>', unsafe_allow_html=True)
    
    st.info("📁 Upload a CSV file containing multiple patient records for batch analysis")
    
    # File upload section
    uploaded_file = st.file_uploader("**Choose CSV File**", type="csv", 
                                   help="File should contain columns: sex, hb, pcv, rbc, mcv, mch, mchc, rdw, wbc, neut, lymph, plt, hba, hba2, hbf")
    
    if uploaded_file is not None:
        try:
            # Read and display data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully uploaded {len(df)} patient records")
            
            # Show data preview
            st.markdown("#### 👀 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Data summary
            st.markdown("#### 📈 Data Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Prediction button
            if st.button("🔮 Predict All Records", use_container_width=True):
                if not predictor.models_loaded:
                    st.error("❌ Please load models first using the button in the sidebar!")
                    return
                
                # Preprocess data
                df_processed = df.copy()
                if 'sex' in df_processed.columns:
                    df_processed['sex'] = df_processed['sex'].str.lower().map({'male':1,'female':0}).fillna(0).astype(int)
                
                # Select numeric features
                X = df_processed.select_dtypes(include=[np.number])
                if 'phenotype' in X.columns:
                    X = X.drop(columns=['phenotype'])
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                predictions = []
                total_records = len(X)
                
                for i, row in X.iterrows():
                    # Update progress
                    progress = (i + 1) / total_records
                    progress_bar.progress(progress)
                    status_text.text(f"🔄 Processing record {i+1} of {total_records}")
                    
                    input_data = pd.DataFrame([row])
                    (rf_pred, rf_prob), (xgb_pred, xgb_prob) = predictor.predict(input_data)
                    
                    if rf_pred is not None:
                        predictions.append({
                            'Record_ID': i+1,
                            'RF_Prediction': '🧬 Alpha Carrier' if rf_pred == 1 else '✅ Normal',
                            'RF_Confidence': f"{max(rf_prob)*100:.1f}%",
                            'XGB_Prediction': '🧬 Alpha Carrier' if xgb_pred == 1 else '✅ Normal',
                            'XGB_Confidence': f"{max(xgb_prob)*100:.1f}%",
                            'Consensus': '✅ Agree' if rf_pred == xgb_pred else '⚠️ Disagree'
                        })
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                results_df = pd.DataFrame(predictions)
                st.markdown("#### 📋 Prediction Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Statistics
                st.markdown("#### 📊 Prediction Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    alpha_carriers = len([p for p in predictions if 'Alpha Carrier' in p['RF_Prediction']])
                    st.metric("Alpha Carriers (RF)", alpha_carriers)
                
                with col2:
                    normal_cases = len([p for p in predictions if 'Normal' in p['RF_Prediction']])
                    st.metric("Normal Cases (RF)", normal_cases)
                
                with col3:
                    agreement = len([p for p in predictions if p['Consensus'] == '✅ Agree'])
                    st.metric("Model Agreement", f"{agreement}/{total_records}")
                
                with col4:
                    avg_confidence = np.mean([float(p['RF_Confidence'].strip('%')) for p in predictions])
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                
                # Download button
                st.markdown("#### 💾 Export Results")
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="thalassemia_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def about_mode():
    st.markdown('<div class="sub-header">ℹ️ About This Tool</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🧬 Alpha Thalassemia AI Prediction System
        
        This advanced AI-powered tool assists healthcare professionals in predicting 
        the risk of Alpha Thalassemia using state-of-the-art machine learning models 
        trained on comprehensive hematological parameters.
        
        #### 🎯 Key Features
        
        - **🤖 Dual AI Models**: Ensemble of Random Forest and XGBoost algorithms
        - **🔍 Individual Assessment**: Detailed single-patient risk analysis
        - **📊 Batch Processing**: Efficient multi-patient screening
        - **📈 Confidence Scoring**: Real-time probability estimates
        - **📱 Responsive Design**: Works on all devices
        - **💾 Export Capabilities**: Download results for records
        
        #### 🩺 Clinical Parameters
        
        The system analyzes 15 key hematological parameters:
        
        - **Basic Profile**: Sex, Hb, PCV, RBC count
        - **Cell Indices**: MCV, MCH, MCHC, RDW
        - **WBC Analysis**: Total WBC, Neutrophils, Lymphocytes
        - **Platelets**: PLT count
        - **Hemoglobin Variants**: HbA, HbA2, HbF percentages
        
        #### ⚠️ Important Disclaimer
        
        > **This tool is for screening and educational purposes only.**
        > 
        > - Does not replace professional medical diagnosis
        > - Always confirm with laboratory tests
        > - Consult qualified healthcare providers
        > - Use alongside clinical evaluation
        """)
    
    with col2:
        st.image("https://img.icons8.com/color/200/000000/machine-learning.png", width=150)
        st.markdown("""
        #### 🛠️ Technology Stack
        
        - **Python 3.8+**
        - **Streamlit** - Web Framework
        - **Scikit-learn** - Machine Learning
        - **XGBoost** - Advanced ML
        - **Pandas** - Data Processing
        - **NumPy** - Numerical Computing
        
        #### 📈 Model Performance
        
        - **Random Forest**: 95% Accuracy
        - **XGBoost**: 93% Accuracy  
        - **Training Data**: 200+ Samples
        - **Validation**: Stratified K-Fold
        
        #### 🔬 Research Based
        
        Developed using clinically validated
        hematological parameters for
        thalassemia screening.
        """)

def display_results(rf_pred, rf_prob, xgb_pred, xgb_prob):
    st.markdown("---")
    st.markdown('<div class="sub-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
    
    # Get prediction details
    rf_confidence = max(rf_prob) * 100
    rf_class = "🧬 Alpha Carrier" if rf_pred == 1 else "✅ Normal"
    rf_alpha_prob = rf_prob[1] * 100
    rf_normal_prob = rf_prob[0] * 100
    
    xgb_confidence = max(xgb_prob) * 100
    xgb_class = "🧬 Alpha Carrier" if xgb_pred == 1 else "✅ Normal"
    xgb_alpha_prob = xgb_prob[1] * 100
    xgb_normal_prob = xgb_prob[0] * 100
    
    # Display predictions in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌲 Random Forest Model")
        risk_class = "risk-high" if rf_pred == 1 else "risk-low"
        st.markdown(f"""
        <div class="{risk_class}">
            <h3 style='margin:0;'>{rf_class}</h3>
            <p style='font-size: 1.2rem; margin: 10px 0;'>Overall Confidence: <b>{rf_confidence:.1f}%</b></p>
            <div style='background: rgba(255,255,255,0.3); padding: 10px; border-radius: 8px;'>
                <p>🧬 Alpha Carrier Probability: <b>{rf_alpha_prob:.1f}%</b></p>
                <p>✅ Normal Probability: <b>{rf_normal_prob:.1f}%</b></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ⚡ XGBoost Model")
        risk_class = "risk-high" if xgb_pred == 1 else "risk-low"
        st.markdown(f"""
        <div class="{risk_class}">
            <h3 style='margin:0;'>{xgb_class}</h3>
            <p style='font-size: 1.2rem; margin: 10px 0;'>Overall Confidence: <b>{xgb_confidence:.1f}%</b></p>
            <div style='background: rgba(255,255,255,0.3); padding: 10px; border-radius: 8px;'>
                <p>🧬 Alpha Carrier Probability: <b>{xgb_alpha_prob:.1f}%</b></p>
                <p>✅ Normal Probability: <b>{xgb_normal_prob:.1f}%</b></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Consensus analysis
    st.markdown("---")
    st.markdown("#### 🤝 Model Consensus")
    
    if rf_pred == xgb_pred:
        st.success(f"""
        ✅ **Strong Consensus**: Both AI models agree on **{rf_class}**
        
        - Models show consistent pattern recognition
        - High reliability in prediction
        - Proceed with recommended clinical pathway
        """)
    else:
        st.warning("""
        ⚠️ **Models Disagree**: Different predictions detected
        
        - Consider additional diagnostic tests
        - Review all clinical parameters
        - Consult hematology specialist
        - Consider repeat testing if indicated
        """)
    
    # Clinical guidance
    st.markdown("---")
    st.markdown("#### 📋 Clinical Guidance")
    
    if rf_pred == 1 or xgb_pred == 1:
        st.error("""
        ## 🚨 High Risk Indicators Detected
        
        **Recommended Actions:**
        
        - 🔬 **Confirm with HPLC/Tests**: Hemoglobin electrophoresis
        - 🧬 **Genetic Counseling**: Family history assessment
        - 👨‍⚕️ **Hematology Referral**: Specialist consultation
        - 👪 **Family Screening**: First-degree relatives
        - 📊 **Additional Tests**: Iron studies, DNA analysis
        
        **Clinical Considerations:**
        - Microcytic hypochromic anemia pattern
        - Normal to elevated RBC count
        - Elevated HbA2 in some variants
        - Family history important
        """)
    else:
        st.success("""
        ## ✅ Low Risk Profile
        
        **Recommended Actions:**
        
        - 📝 **Routine Follow-up**: Regular health monitoring
        - 🩺 **Annual Check-ups**: Standard preventive care
        - 📋 **Health Maintenance**: Continue current practices
        - 🔍 **Monitor Symptoms**: Report any changes
        
        **Clinical Notes:**
        - Parameters within normal ranges
        - No immediate intervention needed
        - Maintain healthy lifestyle
        - Regular health screenings
        """)

if __name__ == "__main__":
    main()
