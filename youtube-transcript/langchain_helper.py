from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
import re
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets["openai_api_key"]
    except:
        raise ValueError("OpenAI API key not found in environment variables or Streamlit secrets")

os.environ["OPENAI_API_KEY"] = api_key

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube URL")

def create_vector_db(youtube_video_url: str) -> FAISS:
    video_id = extract_video_id(youtube_video_url)
    ytt_api = YouTubeTranscriptApi()
    
    try:
        fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
        
        transcript_text = ""
        for snippet in fetched_transcript:
            transcript_text += snippet.text + " "
        
        document = Document(page_content=transcript_text.strip())
        
    except Exception as e:
        try:
            transcript_list = ytt_api.list(video_id)
            transcript = transcript_list.find_transcript(['en'])
            fetched_transcript = transcript.fetch()
            
            transcript_text = ""
            for snippet in fetched_transcript:
                transcript_text += snippet.text + " "
            
            document = Document(page_content=transcript_text.strip())
        except Exception as fallback_error:
            raise Exception(f"Could not fetch transcript: {e}. Fallback also failed: {fallback_error}")

    # Optimized chunking for GPT-5-nano's large context window
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, 
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
  
    docs = text_splitter.split_documents([document])

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db: FAISS, query: str) -> str:
    docs = db.similarity_search(query, k=8)
    docs_page_content = "\n".join([doc.page_content for doc in docs])
    llm = ChatOpenAI(model="gpt-5-nano")
    prompt = PromptTemplate(
        template="""
        You are a helpful YouTube assistant that can answer questions about videos based on the video's transcript.
        
        Instructions:
        - Answer ONLY from the provided context below
        - If the information is not available in the context, say "I don't know"
        - Provide comprehensive, well-structured answers
        - Use markdown formatting for better readability:
          * Use **bold** for section headers
          * Use bullet points (-) for lists
          * Use numbered lists (1., 2., 3.) for sequential items
          * Use line breaks between sections
          * Use *italics* for emphasis
        - Include relevant details and examples when available
        - If discussing multiple topics, organize your response clearly
        - Answer in the same language as the question
        
        Question: {query}
        
        Context from video transcript:
        {docs_page_content}
        
        Answer:
        """,
        input_variables=["query", "docs_page_content"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"query": query, "docs_page_content": docs_page_content})
    return response["text"]