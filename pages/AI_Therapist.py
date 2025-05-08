import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from PIL import Image
import base64
from io import BytesIO


OLLAMA_MODEL = "Koesn/llama3-8b-instruct"

st.set_page_config(page_title="AI Therapist", page_icon="🧠", layout="centered")

st.title("🧠 AI Therapist")
def logo_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

try:
    logo = Image.open("assets/therapy.jpg")
    st.markdown(
        f"""
        <br>
        <br>
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{logo_to_base64(logo)}" width="400"/>
        </div>
        <br>
        <br>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.warning("Logo image not found. Place 'therapy.jpg' in the 'assets' folder.")

st.markdown("""
Welcome. I'm here to listen and support you.  
Feel free to share what's on your mind.

**Disclaimer:** This is an AI application for supportive listening and reflection.  
It is not a substitute for professional therapy or medical advice.  
If you are in distress, please consult a qualified healthcare provider.
""")

st.info("Ensure the Ollama server is running for the therapist to function.")
st.divider()



@st.cache_resource
def load_llm():
    try:
        return OllamaLLM(model=OLLAMA_MODEL)
    except Exception as e:
        st.error(f"Failed to connect to Ollama model '{OLLAMA_MODEL}'. Ensure Ollama is running. Error: {e}")
        return None

llm = load_llm()

therapist_prompt = ChatPromptTemplate.from_template("""
You are a compassionate, thoughtful, and supportive AI therapist.
Your role is to make people feel heard, understood, and comforted.
Use gentle, kind language — keep responses short and empathetic, as if you're in a real therapy session.

Always speak in the tone of a calm, professional therapist — warm, encouraging, and never robotic.

Here is the ongoing conversation:
{chat_history}

User: {user_input}
Therapist:""")

report_prompt_template = ChatPromptTemplate.from_template("""
You are an AI assistant reviewing a transcript of a supportive conversation.
Your task is to provide a reflective summary for the user.

Conversation History:
{chat_history}

Based *only* on the conversation above, please generate a gentle summary.
1.  Start by clearly stating: "This is an AI-generated reflection based on our conversation and is NOT a medical diagnosis. It's intended to offer a perspective on the themes we discussed. For any health concerns, please consult a qualified healthcare professional."
2.  Identify any recurring topics, key emotions, or challenges the user expressed.
3.  If certain patterns of thought or feeling were prominent, you can *gently* point these out.
4.  **DO NOT use diagnostic terms or suggest any specific "disease" or disorder.**
5.  Maintain a supportive, empathetic, and non-judgmental tone.
6.  Conclude by reiterating the importance of seeking professional advice.

Reflective Summary:
""")

if "therapist_memory" not in st.session_state:
    st.session_state.therapist_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="text"
    )

if "therapist_messages" not in st.session_state:
    st.session_state.therapist_messages = []

if "therapist_report_generated" not in st.session_state:
    st.session_state.therapist_report_generated = False

@st.cache_resource
def get_conversation_chain(_llm, _prompt, _memory):
    if _llm is None:
        return None
    return LLMChain(llm=_llm, prompt=_prompt, memory=_memory, verbose=False)

@st.cache_resource
def get_report_chain(_llm, _prompt):
    if _llm is None:
        return None
    return LLMChain(llm=_llm, prompt=_prompt, verbose=False)

if llm:
    conversation_chain = get_conversation_chain(llm, therapist_prompt, st.session_state.therapist_memory)
    report_generating_chain = get_report_chain(llm, report_prompt_template)
else:
    conversation_chain = None
    report_generating_chain = None

for msg in st.session_state.therapist_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.therapist_report_generated and llm and conversation_chain:
    user_input = st.chat_input("How are you feeling today?")
    if user_input:
        st.session_state.therapist_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    full_response = conversation_chain.invoke({"user_input": user_input})
                    therapist_reply = full_response["text"].strip()
                    st.markdown(therapist_reply)
                    st.session_state.therapist_messages.append({"role": "assistant", "content": therapist_reply})
                except Exception as e:
                    st.error(f"Error communicating with the AI model: {e}")
elif st.session_state.therapist_report_generated:
    st.info("Session ended. A summary is shown below. To begin a new session, use the button at the bottom.")
elif not llm:
    st.warning("AI Therapist is currently unavailable. Please check Ollama server.")

st.divider()

if llm and report_generating_chain:
    if not st.session_state.therapist_messages:
        st.markdown("Start a conversation to generate a session summary.")
    elif not st.session_state.therapist_report_generated:
        if st.button("End Session & Get Reflective Summary"):
            if not st.session_state.therapist_memory.chat_memory.messages:
                st.warning("No conversation to summarize yet.")
            else:
                st.session_state.therapist_report_generated = True
                st.subheader("Reflective Session Summary")
                with st.spinner("Generating your summary..."):
                    chat_history_for_report = st.session_state.therapist_memory.load_memory_variables({})["chat_history"]
                    formatted_history = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history_for_report])
                    try:
                        report_response = report_generating_chain.invoke({"chat_history": formatted_history})
                        report_text = report_response["text"].strip()
                        st.markdown(report_text)
                        st.session_state.last_therapist_report_text = report_text
                    except Exception as e:
                        st.error(f"Error generating the summary: {e}")
                        st.session_state.therapist_report_generated = False
    elif "last_therapist_report_text" in st.session_state:
        st.subheader("Reflective Session Summary")
        st.markdown(st.session_state.last_therapist_report_text)

if st.button("Start New Therapist Session (Clear Chat & Summary)"):
    st.session_state.therapist_messages = []
    if "therapist_memory" in st.session_state:
        st.session_state.therapist_memory.clear()
    st.session_state.therapist_report_generated = False
    if "last_therapist_report_text" in st.session_state:
        del st.session_state.last_therapist_report_text
    st.rerun()
