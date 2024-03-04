import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

#######
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from PIL import Image

import os
 


os.environ["OPENAI_API_KEY"] = "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"

# Sidebar contents

import streamlit as st

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #373641;
    }
    .sidebar-content {
        color: #ffffff; 
    }
</style>
            
""", unsafe_allow_html=True)
# Sidebar contents
with st.sidebar:
    chatbot_img = Image.open("img/talkclass.png")
    st.image(chatbot_img, width=150)
    st.markdown("""
    <style>
        img {
            max-height: 300px;
            margin-bottom: 30px;
            margin-left: 73px;
        }
    </style>
    """, unsafe_allow_html=True)

    title = '<div style="text-align: center;"> \
                <p style="font-family:sans-serif; font-weight: bold; color:white; font-size: 42px;">Chatbot</p> \
                <p style="font-family:sans-serif; color:white; font-size: 20px;">에게 물어봐</p> \
                <p style="font-family:sans-serif; color:white; font-size: 15px;">시스템을 이용하는데 <br> 불편한 점이 있나요?</p> \
                <p style="font-family:sans-serif; color:white; font-size: 15px;">고객센터에 문의하기 전에 <br> 저한테 먼저 한 번 물어보세요.</p> \
            </div>'

    st.markdown(title, unsafe_allow_html=True)


def main():
    openai = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0.1,
)
 
    # PDF 업로드
    main_title= '<p style="font-family:sans-serif; color:black; font-weight: bold; font-size: 42px;">Chatbot.  <span style="font-family:sans-serif; font-size: 18px;">에게 물어봐</span></p>'
    st.markdown(main_title, unsafe_allow_html=True)

    pdf = st.file_uploader("Upload your PDF", type='pdf')

    # st.write(pdf)


    
    if pdf is not None:
        # 업로드된 파일의 이름을 얻음
        pdf_name = pdf.name

        # 파일을 저장할 임시 디렉토리 경로 설정
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)

        # 파일 저장
        pdf_path = os.path.join(temp_dir, pdf_name)
        with open(pdf_path, "wb") as f:
            f.write(pdf.read())
        
        # PyPDFLoader 초기화
        loader = PyPDFLoader(pdf_path)
        
        pages = loader.load_and_split()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        docs = text_splitter.split_documents(pages)

        # embeddings
        model_name = "jhgan/ko-sbert-nli"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        persist_directory = "./chroma_db"

        vectorstore = Chroma.from_documents(docs, hf, persist_directory=persist_directory)
        
        # store_name = pdf.name[:-4]
        # st.write(f"Upload File: {store_name}") 
        # st.write(f"Chunks: {docs}")

        # Accept user questions/query
        query = st.text_input("Ask question about your PDF file:")

        st.markdown(f"<div style='border-radius: 20px; padding: 10px; backgroundColor: #c9d4de;'>{query}</div>", unsafe_allow_html=True)

        if query:
            with st.spinner("Searching for answer..."):
                # Retrieval QA
                QA_CHAIN_PROMPT_TEMPLATE = """당신은 이름은 ChatPDF야 업로드된 PDF에 관한 내용만 답할 수 있어 \
                    당신이 하는 일은 업로드된 PDF파일에 대해 구체적이고 친절하게 답하는 일이야 \
                    업로드한 PDF에 관한 질문이 아닌 네가 원래 알고 있었던 지식에 대해선 답할 수 없어 \
                    당신은 업로드한 PDF에 관한 지식이 풍부하고 PDF에만 나와있는 내용에 대해 답변을 명확하게 할 수 있습니다. \
                    사용자가 질문하지 않을 경우에는 자세한 질문을 요구해도되고 PDF에 관한 질문이 아닌 경우에는 답할 수 없어.
                {context}
                질문: {question}
                질문에 대한 답변: """

                QA_CHAIN_PROMPT = PromptTemplate.from_template(QA_CHAIN_PROMPT_TEMPLATE)

                qa = RetrievalQA.from_chain_type(
                    llm=openai,
                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                    retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 1, 'fetch_k': 2}),
                    return_source_documents=True,
                )

                result = qa(query) 
                file_source = result["source_documents"][0].metadata['source']
                file_name = file_source.split("/")[2]
                name = file_name.split(".")

                st.markdown(f"<div style='margin-top: 10px; border-radius: 20px; padding: 10px; backgroundColor: #e9cfd0;'>{result['result']} <br>출처: {name[0]}</div>", unsafe_allow_html=True)
      


if __name__ == '__main__':
    main()
