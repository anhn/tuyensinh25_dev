import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# OpenAI API Key (set in the environment)
#openai.api_key = st.secrets["api"]["key"]
os.environ["OPENAI_API_KEY"] = st.secrets["api"]["key"]
client = OpenAI()

# Sample FAQ database
faq_data = [
    {"question": "What is the admission process?", "answer": "The admission process includes submitting an application, transcripts, and meeting the eligibility criteria."},
    {"question": "What are the tuition fees?", "answer": "Tuition fees vary by program. Visit our tuition fee page for details."},
    {"question": "How do I apply for a scholarship?", "answer": "Scholarships are available for qualified students. Check our scholarship page for eligibility."},
    {"question": "What is the deadline for applications?", "answer": "Application deadlines vary by program and intake. Please check the admissions page for exact dates."}
]

# Convert FAQ questions to embeddings
faq_questions = [item["question"] for item in faq_data]
faq_embeddings = sbert_model.encode(faq_questions, convert_to_tensor=True).cpu().numpy()

# Build FAISS index
faiss_index = faiss.IndexFlatL2(faq_embeddings.shape[1])
faiss_index.add(faq_embeddings)

# Function to find best match using SBERT
def find_best_match(user_query):
    query_embedding = sbert_model.encode([user_query], convert_to_tensor=True).cpu().numpy()
    _, best_match_idx = faiss_index.search(query_embedding, 1)
    return faq_data[best_match_idx[0][0]]

# Function to generate GPT-4 response
def generate_gpt4_response(question, context):
    prompt = (
        f"You are a helpful university admissions assistant.\n"
        f"A student asked: {question}\n\n"
        f"Based on the university information below, provide a helpful, concise, and friendly response:\n\n"
        f"FAQ Answer: {context}\n\n"
        f"Response:"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful university admissions assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Streamlit UI
st.title("🎓 University Admissions Chatbot")
st.write("Ask me anything about university admissions!")

user_input = st.text_input("Type your question here:")

if user_input:
    # Find best FAQ match
    best_match = find_best_match(user_input)
    
    # Generate GPT-4 response
    final_response = generate_gpt4_response(user_input, best_match["answer"])

    st.subheader("🤖 Chatbot Response")
    st.write(final_response)
    
    st.subheader("📌 Matched FAQ")
    st.write(f"**Q:** {best_match['question']}")
    st.write(f"**A:** {best_match['answer']}")
