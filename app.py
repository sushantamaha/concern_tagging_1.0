import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Concern Tagging",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("Mental Health Concern Tagging")
st.write("Please provide your concern or message, and we will analyze it to identify and tag any relevant mental health-related patterns or concerns.")

# Initialize the Google GenerativeAI model
@st.cache_resource
def initialize_model():
    return ChatGoogleGenerativeAI(
        model='gemini-pro',
        temperature=0.5,
        api_key=st.secrets["GOOGLE_API_KEY"]  # Store your API key in Streamlit secrets
    )

# Define prompts dictionary
prompts = {
    "anxiety": "Does the following text discuss feelings of worry, nervousness, unease, or express concerns about future events? Does it mention physical symptoms like rapid heartbeat, sweating, or difficulty breathing? Answer 'True' if yes, 'False' if no.\nText: {text}",
    "depression": "Does the following text express persistent feelings of sadness, hopelessness, loss of interest, or decreased motivation? Does it mention changes in sleep, appetite, or energy levels? Answer 'True' if yes, 'False' if no.\nText: {text}",
    "overthinking": "Does the following text indicate excessive analysis, rumination, or getting stuck in thought loops? Does it show signs of overanalyzing situations or inability to make decisions due to excessive thinking? Answer 'True' if yes, 'False' if no.\nText: {text}",
    "stress": "Does the following text describe feeling overwhelmed, under pressure, or experiencing difficulty coping with demands? Does it mention physical or emotional tension? Answer 'True' if yes, 'False' if no.\nText: {text}",
    "negative_thinking": "Does the following text show patterns of pessimistic thoughts, self-criticism, or focusing primarily on negative aspects of situations? Answer 'True' if yes, 'False' if no.\nText: {text}",
    "loneliness": "Does the following text express feelings of isolation, disconnection from others, or a desire for more meaningful relationships? Does it discuss social isolation or difficulty connecting with others? Answer 'True' if yes, 'False' if no.\nText: {text}",
    "self_improvement": "Does the following text discuss personal growth, development goals, or efforts to better oneself? Does it mention strategies or plans for improving mental health, habits, or life circumstances? Answer 'True' if yes, 'False' if no.\nText: {text}",
    "anger": "Does the following text express feelings of intense frustration, rage, or irritability? Does it mention difficulty controlling emotions or aggressive thoughts? Answer 'True' if yes, 'False' if no.\nText: {text}",
    "grief": "Does the following text discuss experiences of loss, bereavement, or processing difficult life changes? Does it express emotions related to mourning or dealing with significant losses? Answer 'True' if yes, 'False' if no.\nText: {text}",
    "sleep": "Does the following text mention difficulties with sleep patterns, insomnia, or unusual sleep behaviors? Does it discuss changes in sleep quality or quantity? Answer 'True' if yes, 'False' if no.\nText: {text}",
    "ocd": "Does the following text describe recurring thoughts, compulsive behaviors, or strict routines that feel necessary? Does it mention distress about order, cleanliness, or repeated checking? Answer 'True' if yes, 'False' if no.\nText: {text}",
    "sexual_dysfunction": "Does the following text discuss concerns about sexual health, intimacy issues, or changes in sexual function? Does it mention distress about sexual performance or satisfaction? Answer 'True' if yes, 'False' if no.\nText: {text}",
    "bipolar": "Does the following text describe significant mood swings, periods of unusually high energy alternating with low periods, or dramatic changes in behavior and thinking? Answer 'True' if yes, 'False' if no.\nText: {text}",
    "addiction": "Does the following text discuss struggles with substance use, compulsive behaviors, or difficulty controlling specific activities? Does it mention impact on daily life due to these behaviors? Answer 'True' if yes, 'False' if no.\nText: {text}"
}

def evaluate_condition(text, condition, llm):
    prompt = prompts[condition]
    chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["text"], template=prompt))
    response = chain.run(text=text)
    
    response = response.strip().lower()
    if response == 'true':
        return True
    elif response == 'false':
        return False
    else:
        return 'unable'

def evaluate_report(report_text, llm):
    evaluation_results = {}
    
    # Create a progress bar
    progress_bar = st.progress(0)
    total_conditions = len(prompts)
    
    for idx, condition in enumerate(prompts):
        evaluation_results[condition] = evaluate_condition(report_text, condition, llm)
        # Update progress bar
        progress_bar.progress((idx + 1) / total_conditions)
    
    return evaluation_results

def main():
    llm = initialize_model()
    
    # Create text input area
    user_input = st.text_area(
        "Please enter your thoughts or feelings - :",
        height=200,
        placeholder="Share what's on your mind..."
    )
    
    if st.button("Analyze"):
        if user_input:
            if len(user_input) <= 700 :  
                with st.spinner("Analyzing your text..."):
                    # Get evaluation results
                    
                    results = evaluate_report(user_input, llm)
                    
                    # Convert results to DataFrame for better display
                    df = pd.DataFrame([
                        {"Category": k.replace("_", " ").title(), 
                         "Detected": "Yes" if v == True else "No" if v == False else "Unclear"}
                        for k, v in results.items()
                    ])
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Create three columns for different types of results
                    detected = df[df["Detected"] == "Yes"]["Category"].tolist()
                    not_detected = df[df["Detected"] == "No"]["Category"].tolist()
                    unclear = df[df["Detected"] == "Unclear"]["Category"].tolist()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("### 🔍 Detected Patterns")
                        if detected:
                            for item in detected:
                                st.markdown(f"- {item}")
                        else:
                            st.write("None detected")
                    
                    with col2:
                        st.markdown("### ❌ Not Detected")
                        if not_detected:
                            for item in not_detected:
                                st.markdown(f"- {item}")
                        else:
                            st.write("None")
                    
                    with col3:
                        st.markdown("### ❓ Unclear")
                        if unclear:
                            for item in unclear:
                                st.markdown(f"- {item}")
                        else:
                            st.write("None")
                    
                # Add a disclaimer
                st.markdown("---")
                st.markdown("""
                **Disclaimer**: This analysis is for informational purposes only and should not be considered as professional medical advice. 
                If you're experiencing mental health concerns, please consult with a qualified mental health professional.
                """)
            else :
                st.warning("Please enter less than 500 words.")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
