from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def create_vector_db(youtube_video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(youtube_video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db: FAISS, query: str) -> str:
    docs = db.similarity_search(query, k=4)
    docs_page_content = "\n".join([doc.pageContent for doc in docs])
    llm = OpenAI(model="gpt-5-nano")
    prompt = PromptTemplate(
        template="""
        You are a helpful YouTube assisant that can answer questions about videos based on video's transcript.
        Answer ONLY from the provided context. If unsure, say "I don't know".
        Answer the following question: {query}
        By searching the transcript of the video: {docs_page_content}
        Your answers should be concise and to the point.
        Answer in the same language as the question.
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke(query)
    return response.replace("\n", "")
