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
st.write("Please provide your concern or message, and we will analyze it to identify and tag any relevant mental health-related patterns or concerns from this list - Anxiety, Depression, Overthinking, Stress, Negative Thinking, Loneliness, Self Improvement, Anger, Grief, Sleep, Ocd, Sexual Dysfunction, Bipolar, Addiction.")

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
   "anger": "Does the following text express feelings of anger, frustration, irritation, or hostility? Does it mention losing control, feeling enraged, or reacting intensely to situations? Respond with 'True' if yes, 'False' if no.\nText: {text}",
   "grief": "Does the following text express feelings of sadness, loss, mourning, or dealing with major life changes such as the death of a loved one? Does it mention emotions like yearning, longing, or struggling to cope with the absence of someone or something important? Answer 'True' if yes, 'False' if no.\nText: {text}",
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



def display_metric_rows(detected_concerns):
    # Define number of columns per row
    cols_per_row = 6
    
    # Calculate number of rows needed
    num_rows = (len(detected_concerns) + cols_per_row - 1) // cols_per_row
    
    # Create rows and add metrics
    for i in range(num_rows):
        # Create columns for this row
        cols = st.columns(cols_per_row)
        
        # Fill columns with metrics
        for j in range(cols_per_row):
            idx = i * cols_per_row + j
            if idx < len(detected_concerns):
                concern = detected_concerns[idx]
                with cols[j]:
                    st.markdown(f"""
                        <div style="
                            background-color: #4CAF50;
                            color: white;
                            padding: 1rem;
                            border-radius: 10px;
                            text-align: center;
                            margin: 0.5rem 0;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            cursor: pointer;
                            transition: transform 0.2s;
                            hover: transform: scale(1.02);
                        ">
                            <h3 style="margin: 0; font-size: 1.1rem;">{concern}</h3>
                        </div>
                    """, unsafe_allow_html=True)

def main():
    llm = initialize_model()
    
    user_input = st.text_area(
        "Feel free to share your thoughts. Just remember, on this exclusive free plan, we can only process one question per minute—don’t overwhelm us! ",
        height=200,
        placeholder="Share what's on your mind..."
    )
    
    if st.button("Analyze"):
        if user_input:
            if len(user_input) <= 1000:
                with st.spinner("Analyzing your text..."):
                    results = evaluate_report(user_input, llm)
                    detected = [k.replace("_", " ").title() for k, v in results.items() if v == True]
                    
                    st.subheader("Analysis Results")
                    
                    if detected:
                        display_metric_rows(detected)
                    else:
                        st.write("No concerns detected.")
                    
                    st.markdown("---")
                    st.markdown("""
                    **Disclaimer**: This analysis is for informational purposes only and should not be considered as professional medical advice. 
                    If you're experiencing mental health concerns, please consult with a qualified mental health professional.
                    """)
            else:
                st.warning("Please enter less than 1000 words.")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
