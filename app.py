import openai
import nltk
from nltk.tokenize import word_tokenize
from fpdf import FPDF
from docx import Document
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize NLTK
nltk.download('punkt')

# Set OpenAI API key
openai.api_key = ''

# Function to chunk large job descriptions using LangChain
def chunk_text(text, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to extract keywords from each chunk using NLP
def extract_keywords_from_chunks(chunks):
    keywords = []
    for chunk in chunks:
        tokens = word_tokenize(chunk)
        chunk_keywords = [word for word in tokens if word.lower() in ['python', 'machine learning', 'deep learning', 'nlp', 'data wrangling']]
        keywords.extend(chunk_keywords)
    return keywords

# Function to generate professional summary using GPT-4 based on retrieved chunks
def generate_summary_from_chunks(chunks):
    summary = ""
    for chunk in chunks:
        prompt = f"Write a professional summary for a resume using the following information: {chunk}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in writing professional resumes."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        summary += response['choices'][0]['message']['content'].strip() + " "
    return summary.strip()

# Function to generate a PDF resume
def generate_pdf(user_data, summary, education, experience, skills, interests):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(200, 10, txt="Resume - " + user_data['name'], ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Contact Information", ln=True)
    pdf.multi_cell(0, 10, txt=f"Email: {user_data['email']}\nPhone: {user_data['phone']}\nLinkedIn: {user_data['linkedin']}\n")

    # Add Professional Summary
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(200, 10, txt="Professional Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=summary)

    # Add Education
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(200, 10, txt="Education", ln=True)
    pdf.set_font("Arial", size=12)
    for edu in education:
        pdf.multi_cell(0, 10, txt=f"{edu['degree']} in {edu['field']} - {edu['institution']}\nCGPA/Percentage: {edu['cgpa']} | Year of Passing: {edu['year']}\n")
    
    # Add Skills
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(200, 10, txt="Skills", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=", ".join(skills))

    # Add Experience
    if experience:
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, txt="Experience", ln=True)
        pdf.set_font("Arial", size=12)
        for exp in experience:
            pdf.multi_cell(0, 10, txt=f"{exp['role']} at {exp['company']}\nDuration: {exp['duration']}\n")
            for point in exp['responsibilities']:
                pdf.multi_cell(0, 10, txt=f"- {point}")
    
    # Add Interests
    if interests:
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, txt="Interests", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=", ".join(interests))
    
    pdf_output = "generated_resume.pdf"
    pdf.output(pdf_output)
    return pdf_output

# Function to generate a Word resume
def generate_docx(user_data, summary, education, experience, skills, interests):
    doc = Document()
    doc.add_heading(f"Resume - {user_data['name']}", 0)
    doc.add_heading('Contact Information', level=1)
    doc.add_paragraph(f"Email: {user_data['email']}\nPhone: {user_data['phone']}\nLinkedIn: {user_data['linkedin']}")

    doc.add_heading('Professional Summary', level=1)
    doc.add_paragraph(summary)

    doc.add_heading('Education', level=1)
    for edu in education:
        doc.add_paragraph(f"{edu['degree']} in {edu['field']} - {edu['institution']}")
        doc.add_paragraph(f"CGPA/Percentage: {edu['cgpa']} | Year of Passing: {edu['year']}")

    doc.add_heading('Skills', level=1)
    doc.add_paragraph(", ".join(skills))

    if experience:
        doc.add_heading('Experience', level=1)
        for exp in experience:
            doc.add_paragraph(f"{exp['role']} at {exp['company']}")
            doc.add_paragraph(f"Duration: {exp['duration']}")
            for point in exp['responsibilities']:
                doc.add_paragraph(f"- {point}")

    doc.add_heading('Interests', level=1)
    doc.add_paragraph(", ".join(interests))

    doc_output = "generated_resume.docx"
    doc.save(doc_output)
    return doc_output

# Function to generate a Text resume
def generate_txt(user_data, summary, education, experience, skills, interests):
    txt_output = "generated_resume.txt"
    with open(txt_output, 'w') as file:
        file.write(f"Resume - {user_data['name']}\n\n")
        file.write("Contact Information:\n")
        file.write(f"Email: {user_data['email']}\nPhone: {user_data['phone']}\nLinkedIn: {user_data['linkedin']}\n\n")
        file.write("Professional Summary:\n")
        file.write(summary + "\n\n")

        file.write("Education:\n")
        for edu in education:
            file.write(f"{edu['degree']} in {edu['field']} - {edu['institution']}\n")
            file.write(f"CGPA/Percentage: {edu['cgpa']} | Year of Passing: {edu['year']}\n\n")

        file.write("Skills:\n")
        file.write(", ".join(skills) + "\n\n")

        if experience:
            file.write("Experience:\n")
            for exp in experience:
                file.write(f"{exp['role']} at {exp['company']}\n")
                file.write(f"Duration: {exp['duration']}\n")
                for point in exp['responsibilities']:
                    file.write(f"- {point}\n")

        file.write("Interests:\n")
        file.write(", ".join(interests) + "\n")

    return txt_output

# Streamlit App to Collect User Data and Generate Resume
st.title("ATS-Friendly Resume Generator with RAG")

# Input Form for Resume Details
user_name = st.text_input("Full Name")
user_email = st.text_input("Email")
user_phone = st.text_input("Phone")
user_linkedin = st.text_input("LinkedIn Profile URL")
job_description = st.text_area("Job Description (from job post)")

# Education Section (Dynamic)
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

# Work Experience Section (Dynamic)
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

# Skills Section
st.subheader("Skills")
skills = st.text_area("List your skills (comma separated)").split(",")

# Interests Section
st.subheader("Interests")
interests = st.text_area("List your interests (comma separated)").split(",")

# Generate Resume Button
if st.button("Generate Resume"):
    # Process the job description
    chunks = chunk_text(job_description)
    keywords = extract_keywords_from_chunks(chunks)
    summary = generate_summary_from_chunks(chunks)

    user_data = {
        "name": user_name,
        "email": user_email,
        "phone": user_phone,
        "linkedin": user_linkedin
    }

    # Choose Format for Resume
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
