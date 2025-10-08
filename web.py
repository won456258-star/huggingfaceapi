# --- 1. 필수 라이브러리 임포트 ---
import streamlit as st
from dotenv import load_dotenv
import streamlit.components.v1 as components
import os

# LangChain 관련 라이브러리
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# rag_logic.py에서 벡터 DB 생성/로드 함수를 가져옵니다.
from rag_logic import create_and_store_vector_db

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# --- 2. 페이지 설정 및 CSS ---
st.set_page_config(page_title="모구챗 - My RAG 챗봇", page_icon="✨", layout="centered")
st.markdown("""
<style>
    /* ... (이전 CSS와 동일) ... */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Noto Sans KR', sans-serif; }
    .stApp { background: linear-gradient(135deg, #F9F5FF 0%, #E2E1FF 100%); }
    .st-emotion-cache-1f1G203 { background-color: white; border-radius: 1.5rem; padding: 1.5rem; margin: 1rem; box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1); border: 1px solid rgba(255, 255, 255, 0.18); height: 85vh; padding-bottom: 5rem; }
    [data-testid="stChatMessage"][data-testid-role="assistant"] .st-emotion-cache-124el85 { background-color: #F0F0F5; border-radius: 20px 20px 20px 5px; color: #111; border: 1px solid #E5E7EB; animation: fadeIn 0.5s ease-in-out; }
    [data-testid="stChatMessage"][data-testid-role="assistant"] .st-emotion-cache-t3u2ir { background: linear-gradient(45deg, #7A42E2, #9469F4); color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    [data-testid="stChatMessage"][data-testid-role="user"] .st-emotion-cache-124el85 { background: linear-gradient(45deg, #7A42E2, #9469F4); border-radius: 20px 20px 5px 20px; color: white; animation: fadeIn 0.5s ease-in-out; }
    .faq-card { background-color: rgba(249, 245, 255, 0.8); border: 1px solid rgba(255, 255, 255, 0.3); padding: 1.2rem; border-radius: 1rem; margin-bottom: 1rem; }
    .faq-title { font-size: 18px; font-weight: 700; margin-bottom: 1rem; }
    .stButton>button { background-color: #FFFFFF; color: #555; border: 1px solid #DDD; border-radius: 20px; padding: 8px 16px; transition: all 0.2s ease-in-out; box-shadow: 0 1px 2px rgba(0,0,0,0.05); width: 100%; text-align: left; }
    .stButton>button:hover { background-color: #F0F0F5; color: #7A42E2; border-color: #7A42E2; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .stChatInput { background-color: #FFFFFF; padding: 1rem; border-top: 1px solid #E5E7EB; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)


# --- 3. RAG 챗봇 로직 로드 및 체인 구성 (이전과 동일) ---
@st.cache_resource
def get_rag_chain():
    # (이전 코드와 동일하므로 생략)
    if not HUGGINGFACE_API_KEY: return None
    vector_db = create_and_store_vector_db()
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    llm_endpoint = HuggingFaceEndpoint(repo_id="google/gemma-2-9b-it", huggingfacehub_api_token=HUGGINGFACE_API_KEY, temperature=0.3)
    llm = ChatHuggingFace(llm=llm_endpoint)
    system_prompt = """
    당신은 '모구서비스'의 친절하고 정확한 AI 상담원 '모구봇'입니다.
    사용자의 질문에 대해 아래의 '검색된 문서' 내용을 기반으로, '챗봇 대화 가이드'를 참고하여 답변해주세요.
    문서에 없는 내용은 답변할 수 없다고 솔직하게 말해주세요.
    # 챗봇 대화 가이드:
    - 말투: 존댓말이지만 친근한 "~요" 톤
    - 페르소나: 동네메이트처럼 편하지만, 상담원처럼 정확하게 안내
    - 답변 근거: 항상 '검색된 문서'만을 기반으로 답변. 외부 지식 사용 금지.
    - 모르는 질문: "아직 준비되지 않은 정보예요. 곧 업데이트할게요 🙂" 라고 안내
    # 검색된 문서:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return rag_chain

rag_chain = get_rag_chain()


# --- 4. 자동 스크롤 함수 (이전과 동일) ---
def auto_scroll():
    components.html(
        """<script> window.parent.document.querySelector('.st-emotion-cache-1f1G203').scrollTo(0, 99999); </script>""",
        height=0)

# --- 5. UI 렌더링 함수 ---
def render_welcome_elements():
    """추천 질문 UI를 렌더링합니다."""
    
    # 첫 인사 메시지는 채팅 기록이 없을 때만 표시합니다.
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("궁금한 내용을 입력해주시면,\n답변을 빠르게 챗봇이 도와드릴게요.")

    # TOP 3 질문 카드는 항상 표시됩니다.
    st.markdown('<div class="faq-card">', unsafe_allow_html=True)
    st.markdown('<div class="faq-title">많이 찾는 질문 TOP 3</div>', unsafe_allow_html=True)
    
    faq_items = {
        "모구 수수료 제한은 어떻게 되나요?": "💬 모구 수수료 제한",
        "모구 마감 기한은 며칠까지 가능한가요?": "💬 모구 마감 기한",
        "모구에서 팔면 안되는 물건은 무엇인가요?": "💬 모구 판매 금지 품목"
    }
    
    for query, text in faq_items.items():
        if st.button(text, key=query):
            st.session_state.prompt_from_button = query
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# --- 6. 메인 애플리케이션 로직 ---
st.title("모구챗 ✨")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 기록을 표시합니다.
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="✨" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])

# ◀◀◀ 변경점: 버튼 UI 함수를 조건문 밖으로 이동 ◀◀◀
# 추천 질문 UI를 항상 렌더링합니다.
render_welcome_elements()

# 사용자 입력 처리 로직
prompt = st.chat_input("궁금하신 내용을 입력해주세요.")

if "prompt_from_button" in st.session_state and st.session_state.prompt_from_button:
    prompt = st.session_state.prompt_from_button
    st.session_state.prompt_from_button = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✨"):
        if rag_chain:
            response_stream = rag_chain.stream(prompt)
            full_response = st.write_stream(response_stream)
        else:
            full_response = "죄송합니다, 챗봇을 초기화하는 데 문제가 발생했습니다."
            st.write(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    auto_scroll()
    st.rerun()

else:
    auto_scroll()
