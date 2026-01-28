
😄 작성일: 2026-01-28

# RAG 실습 기록 (VS Code 로컬)

폴더: `rag_practice/`  
가상환경: `.venv` 사용  
PDF: `data/` 폴더에 2개 저장  
Chroma DB: `chroma_db/` 생성됨

---

## 실습0) 프로젝트 구조

- `.venv/` : 가상환경
- `data/` : PDF 2개 저장
  - `2040_report.pdf`
  - `OneNYC-2050-Summary.pdf`
- `chroma_db/` : Chroma 벡터DB 저장 폴더(로컬 재사용용)
- `.env` : `OPENAI_API_KEY=...` 저장
- `check_files.py` : data 폴더 PDF 파일명 확인용
- `check_imports.py` : chroma/embeddings 임포트 확인용
- `rag_practice_01.py` : PDF 로딩 + 청킹(Chunking)
- `rag_practice_02.py` : 임베딩 + Chroma DB 생성/로드 + 유사검색
- `rag_practice_03.py` : Retriever + 컨텍스트 기반 답변 생성 + 대화기록 저장
- `rag_practice_04.py` : 질의 확장(Rewrite) → 검색 → 답변 생성

---

## 실습1) PDF 텍스트로 변환 + 청크로 쪼개기

### 1-1. 라이브러리 설치
- 설치한 패키지:
  - `pymupdf`, `pypdf`
  - `langchain_community`
  - `langchain-text-splitters`

### 1-2. PDF 준비
- data 폴더 생성 후 PDF 2개 넣음
  - `OneNYC-2050-Summary.pdf`
  - `2040_report.pdf`

### 1-3. 파일 확인 (check_files.py)
- `data_dir = Path("data")`
- `glob("*.pdf")`로 파일 목록 출력해서 정상 확인

실행 예:
- `python check_files.py`
- 출력: `['2040_report.pdf', 'OneNYC-2050-Summary.pdf']`

### 1-4. PyPDFLoader로 PDF 로딩
- `PyPDFLoader`로 각 PDF를 `Document 리스트`로 로드
- Document는
  - `page_content` : 텍스트
  - `metadata` : page 번호, source 등 포함

확인:
- `len(one_docs), len(seoul_docs)`
- `type(one_docs[0]).__name__`
- `one_docs[0].metadata`

### 1-5. 청킹(Chunking)
- `RecursiveCharacterTextSplitter` 사용
- 오버랩 없는 버전:
  - `chunk_size=1000`, `chunk_overlap=0`
- OneNYC 문서 청킹 결과 확인:
  - `len(one_splits_no)`
  - 앞 2개 chunk 출력 + metadata 확인

### 1-6. all_splits 만들기
- OneNYC + 서울 문서를 합쳐서:
  - `all_splits = one_splits_no + split_documents(seoul_docs)`
- 타입 및 metadata 확인

### 1-7. 서울 문서만 따로 청킹 + 경계 확인
- 오버랩 0인 상태에서 문장 끊김 확인:
  - idx 기준으로 `[-350:]` / `[:350]` 출력

### 1-8. 오버랩 적용 청킹 + 경계 확인
- 오버랩 있는 버전:
  - `chunk_size=1000`, `chunk_overlap=150`
- 오버랩 적용 후 경계가 자연스럽게 이어지는지 확인

---

## 실습2) OpenAI 임베딩 + Chroma 벡터 DB

### 2-1. 라이브러리 설치
- 설치한 패키지:
  - `langchain-chroma`
  - `chromadb`
  - `langchain-openai`

(윈도우에서 chromadb 설치 오류 가능 → Visual C++ Build Tools 필요할 수 있음)

### 2-2. 임포트 확인 (check_imports.py)
- 아래가 에러 없이 임포트되는지 확인
  - `import chromadb`
  - `from langchain_chroma import Chroma`
  - `from langchain_openai import OpenAIEmbeddings`

### 2-3. API 키 준비 (.env)
- `.env`에 `OPENAI_API_KEY` 저장
- 파이썬에서:
  - `load_dotenv()`
  - `assert os.getenv("OPENAI_API_KEY")`

### 2-4. 임베딩 객체 생성
- `OpenAIEmbeddings(model="text-embedding-3-small")`

### 2-5. Chroma DB 생성 + 문서 적재
- `persist_dir = "chroma_db"`
- `collection_name = "rag_docs"`
- `Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=persist_dir, collection_name=collection_name)`

### 2-6. 저장된 DB 로드
- `vectorstore_loaded = Chroma(persist_directory=persist_dir, collection_name=collection_name, embedding_function=embeddings)`

### 2-7. 유사도 검색 테스트
- `similarity_search(query, k=3)`로 관련 chunk 출력
- page metadata 같이 확인

### 2-8. 벡터 길이 확인
- `len(embeddings.embed_query("테스트 문장입니다."))`

---

## 실습3) Retriever + 컨텍스트 기반 답변 생성 (RAG 기본 흐름)

### 3-1. Retriever 만들기
- `retriever = vectorstore_loaded.as_retriever(search_kwargs={"k": 4})`

### 3-2. 검색된 청크 확인
- 질문 넣고 `retriever.invoke(question)`
- chunk 내용 + page 출력

### 3-3. 컨텍스트 문자열로 묶기 (format_docs)
- page/source 메타데이터를 같이 붙여서 컨텍스트 생성

### 3-4. LLM 답변 생성 체인 만들기
- `ChatOpenAI(model="gpt-4o-mini", temperature=0.2)`
- `ChatPromptTemplate` + `StrOutputParser`
- 규칙:
  - 컨텍스트 기반으로만 답변
  - 없으면 "문서에서 확인되지 않았습니다"라고 답변
  - 가능하면 5줄 이내

### 3-5. 질문 → 검색 → 답변 함수
- `ask_with_retrieval(question)`
  - docs 검색
  - context 생성
  - answer_chain으로 답변 생성
  - (answer, docs) 반환

### 3-6. 메시지 저장(chat_history)
- `chat_history = []`
- user/assistant 메시지를 dict로 저장
- 저장된 대화 기록 출력

---

## 실습5) 질의 확장(Query Rewrite)

(실습4는 답변 시스템 프롬프트/체인 구성이라 실습3 코드에 포함해서 진행했음)

### 5-1. StrOutputParser 준비
- LLM 출력이 반드시 "한 줄 질문"으로 나오게 파서 사용

### 5-2. 질문 재작성(구체화) 프롬프트
- 짧고 애매한 질문 → 검색 친화적으로 구체화
- 출력 규칙: 한국어 한 줄 질문 1개만

### 5-3. rewrite_chain 생성
- `rewrite_prompt | rewrite_llm | str_parser`

### 5-4. 원 질문 → 확장 질문 출력
- 예: "도시기본계획 방향이 뭐야?"
- 확장 질문 출력 확인

### 5-5. 확장 질문으로 retriever 검색
- `docs = retriever.invoke(expanded_q)`
- chunk 내용 확인

### 5-6. 확장 질문 + 컨텍스트로 최종 답변 생성
- `context = format_docs(docs)`
- `final_answer = answer_chain.invoke({"question": expanded_q, "context": context})`
- `FINAL ANSWER` 출력

---

## 실행 순서 메모(내 기준)

1) `.venv` 활성화  
2) `python check_files.py` (PDF 확인)  
3) `python rag_practice_01.py` (PDF 로딩 + 청킹)  
4) `python check_imports.py` (chroma 관련 임포트 확인)  
5) `python rag_practice_02.py` (임베딩 + DB 생성/로드 + 검색)  
6) `python rag_practice_03.py` (Retriever + 답변 생성 + chat_history)  
7) `python rag_practice_04.py` (질의 확장 → 검색 → 답변)