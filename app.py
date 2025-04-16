import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

st.set_page_config(layout="wide")

load_dotenv()

# Set your Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)

# Streamlit UI
st.title("📄 JD to Interview Questions Generator (Groq - LLaMA 2 70B)")
st.markdown("Paste the URL of a Jio Careers JD page below:")

# Input box for JD URL
jd_url = st.text_input("Job Description URL")

# Temperature slider
temperature = st.slider("Model Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1)

# Number of questions inputs
tech_qs = st.number_input("Number of Technical Questions", min_value=1, max_value=20, value=5)
behav_qs = st.number_input("Number of Behavioral Questions", min_value=1, max_value=20, value=5)

# Parse JD content from URL
def parse_jio_jd(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        jd_text = soup.get_text(separator='\n')
        return jd_text
    except Exception as e:
        return f"Error parsing JD: {e}"

#def parse_jio_jd(url):
#    try:
#        response = requests.get(url)
#        soup = BeautifulSoup(response.text, 'html.parser')
#        jd_div = soup.find(id='ctl00_ContentPlaceHolder1_lbljobDescription')
#        if jd_div:
#            return jd_div.get_text(separator=' ', strip=True)
#        else:
#            return "❌ Could not find the Job Description content on this page."
#    except Exception as e:
#        return f"Error parsing JD: {e}"


# LangChain Prompt Template
prompt = PromptTemplate(
    input_variables=["job_description", "tech_qs", "behav_qs"],
    template="""
You are an expert interviewer.

Given the following job description, generate a list of {tech_qs} technical and {behav_qs} behavioral interview questions that would help assess a candidate's fit for the role.

Job Description:
{job_description}

Format your response as:
Technical Questions:
1. ...
2. ...

Behavioral Questions:
1. ...
2. ...
"""
)

# Button to trigger question generation
if st.button("Generate Interview Questions"):
    if jd_url:
        jd_content = parse_jio_jd(jd_url)
#        st.subheader("📄 Parsed Job Description")
#        st.text_area("", jd_content, height=300)

        # Setup LLM with selected temperature (Groq LLaMA 2 70B)
        llm = ChatGroq(
            model_name="llama3-70b-8192",groq_api_key=GROQ_API_KEY,
            temperature=temperature
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        with st.spinner("Generating questions..."):
            questions = chain.run({
                "job_description": jd_content,
                "tech_qs": tech_qs,
                "behav_qs": behav_qs
            })
        st.subheader("🧠 Interview Questions")
        #st.text_area("", questions)
        st.write(questions)
    else:
        st.warning("Please enter a valid JD URL.")
