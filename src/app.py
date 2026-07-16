import streamlit as st
import pandas as pd
import joblib
import os

# Set page config for a premium feel
st.set_page_config(
    page_title="Personality AI Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model and vectorizer
MODEL_PATH = 'data/models/personality_model.joblib'
VECTORIZER_PATH = 'data/models/tfidf_vectorizer.joblib'
SAMPLE_DATA_PATH = 'data/raw/fake_resume_dataset.csv'

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        st.error("Model or Vectorizer files not found. Please run the training script first.")
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_models()

# Function to make predictions and convert them to scores out of 10
def predict_personality_traits(resume_text):
    if model is None or vectorizer is None:
        return {}
    resume_tfidf = vectorizer.transform([resume_text])
    predictions = model.predict(resume_tfidf)
    
    # Scale predictions between 0 and 10
    result = {
        "Extroversion": max(0.0, min(round((predictions[0][0] + 1) * 5, 2), 10.0)),
        "Conscientiousness": max(0.0, min(round((predictions[0][1] + 1) * 5, 2), 10.0)),
        "Openness": max(0.0, min(round((predictions[0][2] + 1) * 5, 2), 10.0)),
        "Agreeableness": max(0.0, min(round((predictions[0][3] + 1) * 5, 2), 10.0)),
        "Neuroticism": max(0.0, min(round((predictions[0][4] + 1) * 5, 2), 10.0))
    }
    return result

# Inject premium custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #1e293b;
        font-weight: 800;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
    .trait-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 15px;
        border-left: 5px solid #3b82f6;
    }
    .trait-title {
        font-weight: 700;
        font-size: 1.1rem;
        color: #0f172a;
    }
    .trait-score {
        font-size: 1.8rem;
        font-weight: 800;
        color: #3b82f6;
        float: right;
    }
    .sidebar-header {
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar content: Big Five Model description
with st.sidebar:
    st.markdown('<div class="sidebar-header">🧠 About the OCEAN Model</div>', unsafe_allow_html=True)
    st.markdown("""
    The **Big Five** personality traits (OCEAN) is the standard in modern psychology:
    
    *   **Openness**: Creativity, curiosity, and openness to new experiences.
    *   **Conscientiousness**: Organization, responsibility, and goal-directed behavior.
    *   **Extraversion**: Sociability, assertiveness, and emotional expressiveness.
    *   **Agreeableness**: Trust, altruism, kindness, and affection.
    *   **Neuroticism**: Emotional instability, anxiety, and moodiness.
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ by Antigravity AI")

# Main Header
st.title("🧠 Personality AI Predictor")
st.markdown("##### Analyze resumes to predict Big Five personality traits using Machine Learning.")

# Main tabs
tab1, tab2 = st.tabs(["✍️ Test Single Resume", "📊 Batch Resume Predictor"])

# Tab 1: Single Resume Predictor
with tab1:
    st.write("Paste the text of a resume below to analyze the candidate's personality traits in real-time.")
    
    # Pre-populate with a sample text to make it easy to test
    sample_resume = (
        "Experienced software engineer with a strong background in developing scalable web applications. "
        "Highly collaborative team player with excellent communication skills, leadership experience, "
        "and a passion for learning new technologies."
    )
    
    resume_input = st.text_area("Resume Text", value=sample_resume, height=200)
    
    if st.button("Predict Personality Traits", type="primary"):
        if resume_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                traits = predict_personality_traits(resume_input)
                
                if traits:
                    st.success("Analysis Complete!")
                    
                    # Layout traits in columns
                    cols = st.columns(5)
                    trait_colors = {
                        "Openness": "#3b82f6",          # Blue
                        "Conscientiousness": "#10b981",  # Green
                        "Extroversion": "#f59e0b",       # Orange
                        "Agreeableness": "#8b5cf6",      # Purple
                        "Neuroticism": "#ef4444"         # Red
                    }
                    
                    for idx, (trait, val) in enumerate(traits.items()):
                        color = trait_colors.get(trait, "#3b82f6")
                        with cols[idx]:
                            # Render custom style card for the trait
                            st.markdown(f"""
                            <div class="trait-card" style="border-left-color: {color};">
                                <span class="trait-title">{trait}</span>
                                <div class="trait-score" style="color: {color};">{val}</div>
                                <div style="clear: both; margin-bottom: 10px;"></div>
                            </div>
                            """, unsafe_allow_html=True)
                            # Progress bar
                            st.progress(val / 10.0)

# Tab 2: Batch Predictor
with tab2:
    st.write("Upload a CSV file containing a `Resume_Text` column or use our preloaded sample dataset.")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    
    # Fallback to sample data
    use_sample = False
    if uploaded_file is None:
        st.info("💡 Bypassed file uploader: you can test the application using the preloaded sample dataset.")
        use_sample = st.checkbox("Load Sample Dataset (`fake_resume_dataset.csv`)", value=True)
        
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif use_sample and os.path.exists(SAMPLE_DATA_PATH):
        df = pd.read_csv(SAMPLE_DATA_PATH)
        
    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        if 'Resume_Text' in df.columns:
            st.subheader("Select Resume to Predict")
            
            # Select a resume from dropdown
            resume_options = [f"Row {idx + 1}: {text[:80]}..." for idx, text in enumerate(df['Resume_Text'])]
            selected_idx = st.selectbox("Choose a resume to analyze", range(len(resume_options)), format_func=lambda x: resume_options[x])
            
            selected_resume = df.loc[selected_idx, 'Resume_Text']
            
            st.markdown("### Selected Resume Text:")
            st.info(selected_resume)
            
            # Make prediction
            traits = predict_personality_traits(selected_resume)
            if traits:
                st.markdown("### Predicted Personality Traits:")
                cols = st.columns(5)
                trait_colors = {
                    "Openness": "#3b82f6",
                    "Conscientiousness": "#10b981",
                    "Extroversion": "#f59e0b",
                    "Agreeableness": "#8b5cf6",
                    "Neuroticism": "#ef4444"
                }
                for idx, (trait, val) in enumerate(traits.items()):
                    color = trait_colors.get(trait, "#3b82f6")
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="trait-card" style="border-left-color: {color};">
                            <span class="trait-title">{trait}</span>
                            <div class="trait-score" style="color: {color};">{val}</div>
                            <div style="clear: both; margin-bottom: 10px;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(val / 10.0)
                        
                # If ground truth exists in the CSV, show comparison
                ground_truth_columns = ["Extroversion", "Conscientiousness", "Openness", "Agreeableness", "Neuroticism"]
                if all(col in df.columns for col in ground_truth_columns):
                    st.markdown("### Actual Personality Traits (Ground Truth comparison):")
                    actual_cols = st.columns(5)
                    for idx, trait in enumerate(["Extroversion", "Conscientiousness", "Openness", "Agreeableness", "Neuroticism"]):
                        # Scale database value (assumed -1 to 1 or 0 to 1, we match the database format, but we scale it to match our out of 10 logic)
                        # Looking at fake_resume_dataset.csv, the values are between 0 and 1, so multiply by 10
                        raw_val = df.loc[selected_idx, trait]
                        actual_val = round(raw_val * 10, 2)
                        color = trait_colors.get(trait if trait != "Extroversion" else "Extroversion", "#3b82f6")
                        with actual_cols[idx]:
                            st.markdown(f"""
                            <div class="trait-card" style="border-left-color: #64748b; background-color: #f1f5f9;">
                                <span class="trait-title">{trait} (Actual)</span>
                                <div class="trait-score" style="color: #64748b;">{actual_val}</div>
                                <div style="clear: both; margin-bottom: 10px;"></div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(actual_val / 10.0)
        else:
            st.error("The selected dataset does not contain a 'Resume_Text' column.")

