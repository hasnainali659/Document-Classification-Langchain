from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
    
def load_document(file):
    import os
    name, extension = os.path.splitext(file)
    # file_path = os.path.join(os.getcwd(), 'module5', 'uploads', file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

def chunk_data(data, chunk_size=256):
      
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200) 
    chunks = text_splitter.split_documents(data) 
    return chunks


def standard_retriever(chunks):
    vectordb = Chroma.from_documents(chunks, OpenAIEmbeddings())
    return vectordb
    
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.2)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer

data = load_document('Scopesplit.pdf')
chunks = chunk_data(data, chunk_size=1000)
vectordb = standard_retriever(chunks)

question = """Read Title page, table of content and conclusion and then suggest the category 
this document belongs to from this array [report, letter, handbook, specification document, poem, journal, science book, course outline].
if it is not present in the array then provide a relevant category according to the context.
"""

answer = ask_and_get_answer(vectordb, question)
print(answer)
