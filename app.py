import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import tempfile, os

# -------------------- CONFIG --------------------
st.set_page_config(page_title="TalentScout Hiring Assistant", page_icon="🤖")
st.title("🤖 TalentScout - Intelligent Hiring Assistant")
st.write("Welcome to TalentScout! I’m here to help screen candidates and assess their technical proficiency.")

# -------------------- API KEY --------------------
if "OPENAI_API_KEY" not in os.environ:
    st.warning("Please set your OpenAI API key using: export OPENAI_API_KEY='your_api_key_here'")
    st.stop()

api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)
embeddings = OpenAIEmbeddings(api_key=api_key)

# -------------------- STEP 1: Candidate Info --------------------
st.header("🧾 Step 1: Candidate Information")

with st.form("candidate_form"):
    full_name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    phone = st.text_input("Phone Number")
    experience = st.number_input("Years of Experience", min_value=0, max_value=50)
    position = st.text_input("Desired Position(s)")
    location = st.text_input("Current Location")
    tech_stack = st.text_area("Tech Stack (e.g., Python, Django, React, MySQL)")
    submit_info = st.form_submit_button("Submit Candidate Info")

if submit_info:
    if not (full_name and email and phone and position and tech_stack):
        st.error("⚠️ Please fill in all required fields.")
    else:
        st.success(f"✅ Candidate Info Recorded for {full_name}")
        st.session_state["candidate_info"] = {
            "name": full_name,
            "email": email,
            "phone": phone,
            "experience": experience,
            "position": position,
            "location": location,
            "tech_stack": tech_stack,
        }

# -------------------- STEP 2: Technical Questions --------------------
if "candidate_info" in st.session_state:
    st.header("💻 Step 2: Auto-Generated Technical Questions")

    prompt_template = """
    You are a technical interviewer for a recruitment agency.
    Generate 3 to 5 short technical interview questions based on this candidate’s tech stack:

    Tech Stack: {tech_stack}

    Questions:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["tech_stack"])
    question_chain = LLMChain(llm=llm, prompt=prompt)
    questions = question_chain.run(tech_stack=st.session_state["candidate_info"]["tech_stack"])

    st.write("Here are the generated questions:")
    st.markdown(questions)

# -------------------- STEP 3: Resume Upload --------------------
st.header("📄 Step 3: Resume Upload and Q&A")

uploaded_file = st.file_uploader("Upload Candidate Resume (PDF only)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and process PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()
    resume_text = "\n".join([p.page_content for p in pages])[:4000]  # limit for context window

    st.success("✅ Resume uploaded and processed successfully!")

    # -------------------- STEP 4: Auto Questions from Resume --------------------
    st.header("🤔 Step 4: AI-Generated Questions from Resume")

    resume_question_prompt = PromptTemplate(
        input_variables=["resume_text"],
        template="""
        You are a professional technical interviewer.
        Read the following résumé text and generate 5 to 7 smart interview questions.
        Focus on the candidate’s projects, experience, and technical skills.
        Keep questions short and specific.

        Resume Text:
        {resume_text}

        Questions:
        """,
    )

    resume_question_chain = LLMChain(llm=llm, prompt=resume_question_prompt)
    resume_questions = resume_question_chain.run(resume_text=resume_text)

    st.subheader("🧠 Automatically generated questions based on the résumé:")
    st.markdown(resume_questions)

    # -------------------- STEP 5: Q&A Chat on Resume --------------------
    st.header("💬 Step 5: Chat About the Resume")

    # Create embeddings for resume
    vectorstore = FAISS.from_documents(pages, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question about the uploaded resume (e.g., 'What projects has the candidate done?')")

    if query:
        response = qa_chain({"question": query})
        answer = response["answer"]
        st.session_state.chat_history.append((query, answer))

        st.markdown(f"**Answer:** {answer}")

        with st.expander("Chat History"):
            for q, a in st.session_state.chat_history:
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Assistant:** {a}")

# -------------------- END --------------------
st.divider()
if st.button("End Conversation"):
    st.balloons()
    st.success("🎉 Thank you for using TalentScout! We’ll get back to you with next steps.")
    st.session_state.clear()


