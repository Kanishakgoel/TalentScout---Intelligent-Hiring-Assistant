🤖 TalentScout - Intelligent Hiring Assistant

TalentScout is an AI-powered recruitment assistant built using LangChain, OpenAI, and Streamlit.
It automates the candidate screening process by collecting candidate details, analyzing résumés, and generating context-aware interview questions based on both the candidate’s declared tech stack and their uploaded résumé.

🚀 Features

✅ Smart Candidate Data Collection
Collects essential details such as name, contact info, experience, position, and tech stack.

✅ Automatic Technical Question Generation
Generates 3–5 interview questions tailored to the candidate’s declared tech stack (e.g., Python, Django, React).

✅ Résumé Upload and Analysis (PDF)
Accepts a résumé PDF upload and extracts relevant information for interview preparation.

✅ AI-Generated Questions from Résumé
Automatically generates 5–7 intelligent questions based on the candidate’s projects, experience, and skills from the uploaded résumé.

✅ Interactive Q&A Chat with Résumé
Allows recruiters to chat with the résumé, asking specific questions (e.g., “What are this candidate’s top technical skills?”).

✅ Seamless Conversation Flow
Maintains conversation context and memory for an engaging, human-like interaction.

✅ Easy to Deploy Locally or on Cloud
Runs easily via Streamlit, with optional deployment to platforms like AWS, GCP, or Streamlit Cloud.

🧠 Tech Stack
Component	Description
Python 3.9+	Programming language
Streamlit	Frontend web UI
LangChain	LLM orchestration framework
OpenAI GPT Models	For text generation and question creation
FAISS	Vector store for résumé-based retrieval
PyPDFLoader	For reading and parsing PDF resumes
⚙️ Installation
1️⃣ Clone this repository
git clone https://github.com/yourusername/talentscout-hiring-assistant.git
cd talentscout-hiring-assistant

2️⃣ Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt


(If you don’t have a requirements.txt, create one using:)

pip install streamlit langchain langchain-openai langchain-community faiss-cpu pypdf
pip freeze > requirements.txt

4️⃣ Set your OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"


(On Windows PowerShell:)

setx OPENAI_API_KEY "your_openai_api_key_here"

▶️ Running the App
streamlit run app.py


Then open the displayed local URL (usually http://localhost:8501
) in your browser.

🧩 How It Works

Step 1: The chatbot collects candidate details (name, email, experience, etc.).

Step 2: Based on the declared tech stack, it auto-generates technical questions.

Step 3: The candidate uploads their résumé in PDF format.

Step 4: The system extracts text, analyzes it, and generates interview questions from résumé content.

Step 5: You can chat interactively with the résumé using a retrieval-augmented conversation chain.

📄 Project Structure
TalentScout/
│
├── app.py        # Main Streamlit application           
├── requirements.txt          # Python dependencies              

🌐 Deployment (Optional)

You can deploy this app to:

Streamlit Cloud

🔐 Data Privacy

Candidate data is not stored permanently.

Résumé content is processed in-memory only for Q&A.

Complies with GDPR-like privacy practices.

💬 Example Usage

Tech Stack Input:
Python, Django, React, PostgreSQL

Generated Questions:

1. Explain how Django ORM works internally.  
2. What are React hooks and why are they used?  
3. How do you optimize PostgreSQL queries for performance?


Résumé Questions:

1. Can you describe your project experience using AWS Lambda?  
2. How did you implement CI/CD pipelines in your recent role?

🧑‍💻 Author

Your Name
📧 goelkanishak423@gmail.com
🌐https://www.linkedin.com/in/kanishak-goel-9227a531a/
