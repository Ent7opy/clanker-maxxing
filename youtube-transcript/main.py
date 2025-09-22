import streamlit as st
import langchain_helper as lch

st.title("YouTube Assistant Q&A")

# Initialize session state
if "db" not in st.session_state:
    st.session_state.db = None
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

with st.sidebar:
    with st.expander("YouTube Video Link"):
        youtube_video_url = st.text_input("Enter the YouTube video URL")
        if st.button("Process Video"):
            if youtube_video_url:
                with st.spinner("Processing video..."):
                    st.session_state.db = lch.create_vector_db(youtube_video_url)
                    st.session_state.video_processed = True
                    st.success("Video processed successfully! You can now ask questions.")
            else:
                st.error("Please enter a YouTube URL")

# Main content area
if st.session_state.video_processed and st.session_state.db is not None:
    st.header("Ask Questions About the Video")
    
    # Question input
    question = st.text_input("What would you like to know about this video?", 
                            placeholder="e.g., What is the main topic discussed? What are the key points?")
    
    if st.button("Ask Question") and question:
        with st.spinner("Searching for answer..."):
            response = lch.get_response_from_query(st.session_state.db, question)
            st.write("**Answer:**")
            st.write(response)
    
    # Example questions
    st.subheader("Example Questions:")
    example_questions = [
        "What is the main topic of this video?",
        "What are the key points discussed?",
        "What examples or case studies are mentioned?",
        "What is the conclusion or summary?",
        "What tools or technologies are discussed?"
    ]
    
    for i, example in enumerate(example_questions):
        if st.button(f"Q{i+1}: {example}", key=f"example_{i}"):
            with st.spinner("Searching for answer..."):
                response = lch.get_response_from_query(st.session_state.db, example)
                st.write("**Answer:**")
                st.write(response)

else:
    st.info("👆 Please enter a YouTube video URL in the sidebar and click 'Process Video' to get started!")
    
    # Instructions
    st.markdown("""
    ### How to use this app:
    1. **Enter a YouTube URL** in the sidebar
    2. **Click "Process Video"** to analyze the video transcript
    3. **Ask questions** about the video content
    4. **Get AI-powered answers** based on the video transcript
    
    ### What you can ask:
    - Questions about the main topics
    - Specific details mentioned in the video
    - Summaries of key points
    - Explanations of concepts discussed
    - And much more!
    """)