import streamlit as st
import os
from openai import OpenAI
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
from datetime import datetime
from streamlit_feedback import streamlit_feedback
import requests
import uuid
import time

# Load SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# MongoDB Connection
MONGO_URI = st.secrets["mongo"]["uri"]  # Load MongoDB URI from secrets
PERPLEXITY_API = st.secrets["perplexity"]["key"]
DB_NAME = "utt_detai25"
FAQ_COLLECTION = "faqtuyensinh"
CHATLOG_COLLECTION = "chatlog"

client_mongo = MongoClient(MONGO_URI)
db = client_mongo[DB_NAME]
faq_collection = db[FAQ_COLLECTION]
chatlog_collection = db[CHATLOG_COLLECTION]

def get_ip():
    try:
        return requests.get("https://api64.ipify.org?format=json").json()["ip"]
    except:
        return "Unknown"

user_ip = get_ip()

# Initialize chat history in session state
if "chat_log" not in st.session_state:
    st.session_state["chat_log"] = []

# Set OpenAI API Key in the environment
os.environ["OPENAI_API_KEY"] = st.secrets["api"]["key"]

#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(api_key=PERPLEXITY_API, base_url="https://api.perplexity.ai")


def load_faq_data():
    # Sample FAQ database
    faq_data = [
        {"question": "Quy trình tuyển sinh như thế nào?", "answer": "Quy trình tuyển sinh bao gồm nộp đơn, bảng điểm và đáp ứng các tiêu chí đủ điều kiện."},
        {"question": "Học phí là bao nhiêu?", "answer": "Học phí khác nhau tùy theo chương trình. Vui lòng truy cập trang học phí của chúng tôi để biết chi tiết."},
        {"question": "Làm thế nào để tôi đăng ký học bổng?", "answer": "Học bổng có sẵn cho những sinh viên đủ điều kiện. Hãy kiểm tra trang học bổng của chúng tôi để biết thông tin chi tiết."},
        {"question": "Thời hạn nộp đơn là khi nào?", "answer": "Thời hạn nộp đơn khác nhau tùy theo chương trình và đợt tuyển sinh. Vui lòng kiểm tra trang tuyển sinh để biết ngày cụ thể."}
    ]
    return faq_data
# Convert FAQ questions to embeddings
faq_questions = [item["question"] for item in load_faq_data()]
faq_embeddings = sbert_model.encode(faq_questions, convert_to_tensor=True).cpu().numpy()

# Build FAISS index
faiss_index = faiss.IndexFlatL2(faq_embeddings.shape[1])
faiss_index.add(faq_embeddings)

# Function to find best match using SBERT
def find_best_match(user_query):
    query_embedding = sbert_model.encode([user_query], convert_to_tensor=True).cpu().numpy()
    _, best_match_idx = faiss_index.search(query_embedding, 1)
    best_match = load_faq_data()[best_match_idx[0][0]]
    # Compute similarity
    best_match_embedding = faq_embeddings[best_match_idx[0][0]]
    similarity = util.cos_sim(query_embedding, best_match_embedding).item()
    return best_match, similarity

# Function to generate GPT-4 response
def generate_gpt4_response(question, context):
    prompt = (
        f"Một sinh viên hỏi: {question}\n\n"
        f"Dựa trên thông tin tìm được trên internett, hãy cung cấp một câu trả lời hữu ích, ngắn gọn và thân thiện. Dẫn nguồn nếu có thể."
    )   
    try:
        #response = client.chat.completions.create(  # FIXED API CALL
        #    model="gpt-4",
        #    messages=[
        #        {"role": "system", "content": "Bạn là một trợ lý tuyển sinh đại học hữu ích."},
        #        {"role": "user", "content": prompt}
        #    ],
        #    stream=True
        #)
        response = client.chat.completions.create(
        model="sonar-pro",
        messages=[
                {"role": "system", "content": "Bạn là một trợ lý tuyển sinh đại học hữu ích."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        #return response.choices[0].message.content
        #for message in response:
        #    content = message.choices[0].delta.content
        #    if content:  # Some parts may be None, skip them
        #        yield content
                bot_response = ""  # Store full response
        citations = []  # Store citation sources
        for message in response:
            content = message.choices[0].delta.content
            if content:  # Some parts may be None, skip them
                bot_response += content
            # Extract citations if available
            if "citations" in message.choices[0].delta:
                citations.extend(message.choices[0].delta.citations)
        # Append citations to the response
        if citations:
            bot_response += "\n\n🔗 **Nguồn tham khảo:**\n"
            for citation in citations:
                bot_response += f"- [{citation['title']}]({citation['url']})\n"
        yield bot_response  # Stream the response with citations included

    except Exception as e:
        return f"⚠️ Lỗi: {str(e)}"

# Function to save chat logs to MongoDB
def save_chat_log(user_ip, user_message, bot_response, feedback):
    """Stores chat log into MongoDB, grouped by user IP"""
    if feedback and feedback.strip():
        chat_entry = {
                "user_ip": user_ip,
                "timestamp": datetime.utcnow(),
                "user_message": user_message,
                "bot_response": bot_response,
                "is_good": False,
                "problem_detail": feedback
            }    
    else:    
        chat_entry = {
            "user_ip": user_ip,
            "timestamp": datetime.utcnow(),
            "user_message": user_message,
            "bot_response": bot_response,
            "is_good": True,
            "problem_detail" : ""
        }
    chatlog_collection.insert_one(chat_entry)
    
def stream_text(text):
    """Converts a string into a generator to work with `st.write_stream()`."""
    for word in text.split():
        yield word + " "  # Stream words with a space for a natural effect
        
# Banner Image (Replace with your actual image URL or file path)
BANNER_URL = "https://utt.edu.vn/uploads/images/site/1722045380banner-utt.png"  # Example banner image

st.markdown(
    f"""
    <style>
        .center {{
            text-align: center;
        }}
        .banner {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 450px; /* Adjust size as needed */
        }}
        .title {{
            font-size: 28px;
            font-weight: bold;
            color: #1E88E5; /* Education-themed blue */
            margin-top: 15px;
        }}
        .subtitle {{
            font-size: 18px;
            color: #333;
            margin-top: 5px;
        }}
    </style>

    <div class="center">
        <img class="banner" src="{BANNER_URL}">
        <p class="title">🎓 Hỗ trợ tư vấn tuyển sinh - ĐHCNGTVT</p>
        <p class="subtitle">Hỏi tôi bất kỳ điều gì về tuyển sinh đại học!</p>
    </div>
    """,
    unsafe_allow_html=True
)
# **Chat Interface**
st.subheader("💬 Chatbot Tuyển Sinh")

# **Display Chat History**
for chat in st.session_state["chat_log"]:
    with st.chat_message("user"):
        st.write(chat["user"])
    with st.chat_message("assistant"):
        st.write(chat["bot"])
        
user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Thinking animation
    #with st.chat_message("assistant"):
    #    thinking_container = st.empty()
    #    for _ in range(3):  # Loop to simulate "thinking..."
    #        thinking_container.write("🤖 Chatbot đang suy nghĩ" + "." * (_ + 1))
    #        time.sleep(0.5)  # Pause for effect
    
    # Find best match in FAQ
    best_match, similarity = find_best_match(user_input)
    threshold = 0.7  # Minimum similarity to use FAQ answer
    use_gpt = similarity < threshold

    # Select response source
    if use_gpt:
        response_stream = generate_gpt4_response(user_input, best_match["answer"])  # Now a generator
    else:
        response_stream = stream_text(best_match["answer"])  # FAQ converted to a generator

    # Show bot response in real-time
    with st.chat_message("assistant"):
        bot_response_container = st.empty()  # Create an empty container
        bot_response = ""  # Collect the full response
        for chunk in response_stream:
            bot_response += chunk  # Append streamed content
            bot_response_container.write(bot_response)  # Update UI in real-time

    # Save to session history
    st.session_state["chat_log"].append(
        {"user": user_input, "bot": bot_response, "is_gpt": use_gpt}
    )

    # Save chat log to MongoDB
    save_chat_log(user_ip, user_input, bot_response, "")





