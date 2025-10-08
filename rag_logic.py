# --- 1. 필수 라이브러리 임포트 ---
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 벡터 저장소 파일 경로
VECTOR_DB_PATH = "faiss_index"

# --- 2. 문서 로드 및 분할 함수 (이전과 동일) ---
def load_and_split_document(file_path="my_data.txt"):
    """
    주어진 경로의 텍스트 파일을 로드하고, 의미 있는 단위로 분할합니다.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs

# --- 3. 임베딩 및 벡터 저장소 생성/로드 함수 ---
def create_and_store_vector_db():
    """
    문서를 벡터로 변환하고 FAISS에 저장하거나, 이미 있다면 로드합니다.
    """
    # Hugging Face에서 한국어 임베딩 모델을 가져옵니다.
    # 'jhgan/ko-sroberta-multitask'는 한국어 문맥을 잘 이해하는 모델 중 하나입니다.
    print("⏳ 임베딩 모델을 로드합니다...")
    model_name = "jhgan/ko-sroberta-multitask"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("✅ 임베딩 모델 로드 완료!")

    # 만약 벡터 저장소 파일이 이미 존재하면, 새로 생성하지 않고 바로 로드합니다.
    if os.path.exists(VECTOR_DB_PATH):
        print(f"📂 기존 벡터 저장소 '{VECTOR_DB_PATH}'를 로드합니다.")
        vector_db = FAISS.load_local(
            VECTOR_DB_PATH, 
            embeddings,
            # FAISS 로드 시 필요한 권한 설정입니다.
            allow_dangerous_deserialization=True 
        )
        print("✅ 벡터 저장소 로드 완료!")
        return vector_db

    # 파일이 없다면 새로 생성합니다.
    print(f"✨ 새로운 벡터 저장소를 생성합니다.")
    # 1. 문서를 로드하고 분할합니다.
    docs = load_and_split_document()
    print(f"📄 문서가 성공적으로 로드되었고, {len(docs)}개의 조각으로 분할되었습니다.")

    # 2. 분할된 문서를 기반으로 FAISS 벡터 저장소를 생성합니다.
    print("⏳ 문서를 벡터로 변환하여 저장소에 저장합니다. (시간이 조금 걸릴 수 있습니다...)")
    vector_db = FAISS.from_documents(docs, embeddings)
    
    # 3. 생성된 벡터 저장소를 파일로 저장하여 다음 실행 시 재사용할 수 있게 합니다.
    vector_db.save_local(VECTOR_DB_PATH)
    print(f"✅ 벡터 저장소 생성 및 저장 완료! ('{VECTOR_DB_PATH}' 폴더에 저장됨)")
    
    return vector_db

# --- 4. 메인 실행 블록 (테스트용) ---
if __name__ == '__main__':
    db = create_and_store_vector_db()
    
    # 벡터 저장소가 잘 생성되었는지 테스트하기 위해 간단한 질문으로 검색을 수행합니다.
    query = "모구장이 뭐야?"
    results = db.similarity_search(query, k=2) # 가장 유사한 2개의 결과를 찾습니다.
    
    print(f"\n--- 테스트 검색 결과 (질문: {query}) ---")
    if results:
        for i, doc in enumerate(results):
            print(f"\n[결과 {i+1}]")
            print(f"내용: {doc.page_content[:200]}...") # 내용 일부만 출력
    else:
        print("검색 결과가 없습니다.")
    print("-----------------------------------")