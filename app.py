import openai
import nltk
from nltk.tokenize import word_tokenize
from fpdf import FPDF
from docx import Document
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

@st.cache(allow_output_mutation=True)
def download_nltk_resources():
    nltk.download('punkt')

download_nltk_resources()

# Set the OpenAI API key from Streamlit secrets securely
openai.api_key = st.secrets["apikey"]

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """ Split large text into chunks using the RecursiveCharacterTextSplitter. """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def extract_keywords_from_chunks(chunks):
    """ Extract specified keywords from text chunks. """
    keywords = []
    for chunk in chunks:
        tokens = word_tokenize(chunk)
        # Keywords related to tech and data science
        chunk_keywords = [word for word in tokens if word.lower() in ['python', 'machine learning', 'deep learning', 'nlp', 'data wrangling']]
        keywords.extend(chunk_keywords)
    return keywords

def generate_summary_from_chunks(chunks):
    """ Generate a professional summary using OpenAI GPT-4 based on text chunks. """
    summary = ""
    for chunk in chunks:
        prompt = f"Write a professional summary for a resume using the following information: {chunk}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in writing professional resumes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.0
        )
        summary += response['choices'][0]['message']['content'].strip() + " "
    return summary.strip()

def generate_pdf(user_data, summary, education, experience, skills, interests):
    """ Generate a PDF resume from provided user data and content. """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(200, 10, txt="Resume - " + user_data['name'], ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Contact Information", ln=True)
    pdf.multi_cell(0, 10, txt=f"Email: {user_data['email']}\nPhone: {user_data['phone']}\nLinkedIn: {user_data['linkedin']}\n")

    # Professional Summary
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(200, 10, txt="Professional Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=summary)

    # Education Details
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(200, 10, txt="Education", ln=True)
    pdf.set_font("Arial", size=12)
    for edu in education:
        pdf.multi_cell(0, 10, txt=f"{edu['degree']} in {edu['field']} - {edu['institution']}\nCGPA/Percentage: {edu['cgpa']} | Year of Passing: {edu['year']}\n")
    
    # Skills
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(200, 10, txt="Skills", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=", ".join(skills))

    # Work Experience
    if experience:
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, txt="Experience", ln=True)
        pdf.set_font("Arial", size=12)
        for exp in experience:
            pdf.multi_cell(0, 10, txt=f"{exp['role']} at {exp['company']}\nDuration: {exp['duration']}\n")
            for point in exp['responsibilities']:
                pdf.multi_cell(0, 10, txt=f"- {point}")
    
    # Interests
    if interests:
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, txt="Interests", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=", ".join(interests))
    
    pdf_output = "generated_resume.pdf"
    pdf.output(pdf_output)
    return pdf_output

# Streamlit interface for user input and actions
st.title("ATS-Friendly Resume Generator with RAG")

# Collect user input for resume details
user_name = st.text_input("Full Name")
user_email = st.text_input("Email")
user_phone = st.text_input("Phone")
user_linkedin = st.text_input("LinkedIn Profile URL")
job_description = st.text_area("Job Description (from job post)")

# Dynamic input for education
st.subheader("Education")
education_entries = []
num_education = st.number_input("Number of Education Entries", min_value=1, max_value=10, value=1, step=1)
for i in range(num_education):
    st.write(f"Education {i + 1}")
    degree = st.text_input(f"Degree {i + 1}")
    field = st.text_input(f"Field of Study {i + 1}")
    institution = st.text_input(f"Institution {i + 1}")
    cgpa = st.text_input(f"CGPA/Percentage {i + 1}")
    year = st.text_input(f"Year of Passing {i + 1}")
    education_entries.append({"degree": degree, "field": field, "institution": institution, "cgpa": cgpa, "year": year})

# Dynamic input for work experience
st.subheader("Work Experience")
experience_entries = []
num_experience = st.number_input("Number of Work Experience Entries", min_value=0, max_value=10, value=0, step=1)
for i in range(num_experience):
    st.write(f"Work Experience {i + 1}")
    role = st.text_input(f"Role {i + 1}")
    company = st.text_input(f"Company {i + 1}")
    duration = st.text_input(f"Duration {i + 1}")
    responsibilities = st.text_area(f"Responsibilities {i + 1}")
    responsibilities_list = responsibilities.splitlines() if responsibilities else []
    experience_entries.append({"role": role, "company": company, "duration": duration, "responsibilities": responsibilities_list})

# Input for skills and interests
st.subheader("Skills")
skills = st.text_area("List your skills (comma separated)").split(",")

st.subheader("Interests")
interests = st.text_area("List your interests (comma separated)").split(",")

# Button to generate resume
if st.button("Generate Resume"):
    chunks = chunk_text(job_description)
    keywords = extract_keywords_from_chunks(chunks)
    summary = generate_summary_from_chunks(chunks)

    user_data = {
        "name": user_name,
        "email": user_email,
        "phone": user_phone,
        "linkedin": user_linkedin
    }

    # User selects resume format
    format_choice = st.selectbox("Choose the resume format", ['PDF', 'DOCX', 'TXT'])

    if format_choice == 'PDF':
        pdf_file = generate_pdf(user_data, summary, education_entries, experience_entries, skills, interests)
        st.write(f"Resume generated: {pdf_file}")
        st.download_button("Download PDF", data=open(pdf_file, "rb"), file_name=pdf_file, mime="application/pdf")

    elif format_choice == 'DOCX':
        docx_file = generate_docx(user_data, summary, education_entries, experience_entries, skills, interests)
        st.write(f"Resume generated: {docx_file}")
        st.download_button("Download DOCX", data=open(docx_file, "rb"), file_name=docx_file, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    elif format_choice == 'TXT':
        txt_file = generate_txt(user_data, summary, education_entries, experience_entries, skills, interests)
        st.write(f"Resume generated: {txt_file}")
        st.download_button("Download TXT", data=open(txt_file, "rb"), file_name=txt_file, mime="text/plain")
