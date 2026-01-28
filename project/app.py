import os
import streamlit as st

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG 챗봇", layout="wide")
st.title("RAG 문서 기반 챗봇")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY 환경변수가 필요합니다.")
    st.stop()

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        lines.append(f"(source={src}, page={page}) {d.page_content}")
    return "\n\n".join(lines)

def build_sources(docs):
    items = []
    seen = set()
    for d in docs:
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        items.append({"source": src, "page": page})
    return items

# 1) 질의 확장(재작성) 체인
rewrite_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
rewrite_parser = StrOutputParser()

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "검색용 질문 재작성기입니다. 사용자의 질문을 문서 검색에 적합하도록 더 구체적으로 바꿉니다. "
     "출력은 한 줄의 한국어 질문 1개만 반환합니다."),
    ("user", "원 질문: {q}\n재작성 질문:")
])

rewrite_chain = rewrite_prompt | rewrite_llm | rewrite_parser

# 2) 답변 생성 규칙 강화 prompt
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "문서 기반 질의응답 도우미입니다. 컨텍스트에 포함된 내용만으로 답변합니다. "
     "추정, 상상, 일반상식 보강을 금지합니다. "
     "컨텍스트에 근거 문장이 없으면 '문서에서 확인되지 않았습니다.'라고 답변합니다. "
     "답변은 항목형으로 6줄 이내로 작성합니다."),
    ("user", "질문: {question}\n\n컨텍스트:\n{context}\n\n답변:")
])

# 3) 스트리밍 답변 체인
parser = StrOutputParser()
llm_stream = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)
answer_chain_stream = answer_prompt | llm_stream | parser

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_q = st.chat_input("질문을 입력하세요.")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)

    # 4) 확장 질문 생성 -> 확장 질문으로 검색
    expanded_q = rewrite_chain.invoke({"q": user_q}).strip()

    docs = retriever.invoke(expanded_q)
    context = format_docs(docs)
    sources = build_sources(docs)

    with st.chat_message("assistant"):
        st.caption(f"재작성 질문: {expanded_q}")
        placeholder = st.empty()

        tokens = []
        for chunk in answer_chain_stream.stream({"question": expanded_q, "context": context}):
            tokens.append(chunk)
            placeholder.write("".join(tokens))

        st.write("출처")
        st.dataframe(sources, use_container_width=True)

    answer = "".join(tokens)
    st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption(f"컬렉션 이름: {vectorstore._collection.name}")
st.caption(f"컬렉션 문서 수: {vectorstore._collection.count()}")
st.caption(f"persist 경로(절대): {os.path.abspath(PERSIST_DIR)}")