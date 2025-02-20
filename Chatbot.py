import streamlit as st
import os
from openai import OpenAI
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Set OpenAI API Key in the environment
os.environ["OPENAI_API_KEY"] = st.secrets["api"]["key"]
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)
# Sample FAQ database
faq_data = [
    {"question": "Quy trình tuyển sinh như thế nào?", "answer": "Quy trình tuyển sinh bao gồm nộp đơn, bảng điểm và đáp ứng các tiêu chí đủ điều kiện."},
    {"question": "Học phí là bao nhiêu?", "answer": "Học phí khác nhau tùy theo chương trình. Vui lòng truy cập trang học phí của chúng tôi để biết chi tiết."},
    {"question": "Làm thế nào để tôi đăng ký học bổng?", "answer": "Học bổng có sẵn cho những sinh viên đủ điều kiện. Hãy kiểm tra trang học bổng của chúng tôi để biết thông tin chi tiết."},
    {"question": "Thời hạn nộp đơn là khi nào?", "answer": "Thời hạn nộp đơn khác nhau tùy theo chương trình và đợt tuyển sinh. Vui lòng kiểm tra trang tuyển sinh để biết ngày cụ thể."}
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
        f"Bạn là một trợ lý tuyển sinh đại học hữu ích.\n"
        f"Một sinh viên hỏi: {question}\n\n"
        f"Dựa trên thông tin của trường đại học dưới đây, hãy cung cấp một câu trả lời hữu ích, ngắn gọn và thân thiện:\n\n"
        f"Câu trả lời từ FAQ: {context}\n\n"
        f"Phản hồi:"
    )
    
    try:
        response = client.chat.completions.create(  # FIXED API CALL
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Bạn là một trợ lý tuyển sinh đại học hữu ích."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Lỗi: {str(e)}"

# Streamlit UI
st.title("🎓 University Admissions Chatbot")
st.write("Hỏi tôi bất kỳ điều gì về tuyển sinh đại học!")

user_input = st.text_input("Nhập câu hỏi của bạn:")

if user_input:
    best_match = find_best_match(user_input)
    final_response = generate_gpt4_response(user_input, best_match["answer"])

    st.subheader("🤖 Phản hồi từ chatbot")
    st.write(final_response)
    
    st.subheader("📌 Câu hỏi khớp FAQ")
    st.write(f"**Q:** {best_match['question']}")
    st.write(f"**A:** {best_match['answer']}")
