import streamlit as st
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory  # Fixed import path
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.query_processing import CUSTOM_QUESTION_PROMPT
from backend.pinecone_storage import get_langchain_retriever

# Load environment variables
load_dotenv()

def create_conversation_chain(namespace="default"):
    """
    Create a LangChain conversation chain using Gemini model.
    
    Args:
        namespace: Pinecone namespace
        
    Returns:
        conversation_chain: LangChain ConversationalRetrievalChain
    """
    try:
        # Check if Google API key is set
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("Google API key not found. Please set the GOOGLE_API_KEY in your .env file.")
            return None
        
        # Get retriever
        retriever = get_langchain_retriever(namespace)
        if not retriever:
            st.error("Failed to create retriever")
            return None
        
        # Create Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.2,
            convert_system_message_to_human=True
        )
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        # Create conversation chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            condense_question_prompt=CUSTOM_QUESTION_PROMPT,
            memory=memory,
            return_source_documents=True  # Make sure to return source documents
        )
        
        return conversation_chain
        
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def generate_direct_response_with_chunks(query, chunks):
    """Generate a direct response using retrieved chunks."""
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "I can't provide information without a valid API key."
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )
        
        # Build context from chunks
        context = ""
        for i, chunk in enumerate(chunks):
            if i >= 5:  # Limit to top 5 chunks to avoid token limits
                break
            chunk_text = chunk.get('text', '')
            if not chunk_text:
                continue
            context += f"\nDocument Excerpt {i+1}:\n{chunk_text}\n"
        
        # If no valid chunks were found, return an error
        if not context.strip():
            return "I couldn't extract useful content from the retrieved documents."
        
        prompt = f"""
        Based ONLY on the following information from the documents:
        
        {context}
        
        Provide a comprehensive, well-organized answer to this query: "{query}"
        
        Your response should:
        1. Be well-structured with clear headings if appropriate
        2. Use bullet points for listing strategies, techniques, or steps
        3. Directly answer the query using ONLY the information in the provided document excerpts
        4. NOT include any information not found in the provided excerpts
        5. Be factual and objective
        """
            
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        st.error(f"Error in generate_direct_response_with_chunks: {str(e)}")
        return f"I encountered an error trying to answer your question: {str(e)}"

def generate_response(query, chunks=None):
    """
    Generate a response to the query using LangChain with Gemini.
    
    Args:
        query: User query
        chunks: Retrieved text chunks
        
    Returns:
        response: Generated response
    """
    try:
        # ALWAYS try to use chunks directly if available
        if chunks and len(chunks) > 0:
            return generate_direct_response_with_chunks(query, chunks)
        
        # Only use conversation chain if no chunks were provided
        if 'conversation_chain' not in st.session_state:
            st.session_state.conversation_chain = create_conversation_chain(
                namespace=st.session_state.namespace
            )
        
        if not st.session_state.conversation_chain:
            # If we have chunks but no conversation chain, still try to generate a response
            if chunks and len(chunks) > 0:
                return generate_direct_response_with_chunks(query, chunks)
            return "Failed to create conversation chain. Please check your Google API key."
        
        # Generate response using conversation chain
        with st.spinner("Generating response with Gemini..."):
            response = st.session_state.conversation_chain({'question': query})
            
        # Store the last response in session state
        if 'last_response' not in st.session_state:
            st.session_state.last_response = response
        
        # Extract the answer from the response
        answer = response.get('answer', "No answer found")
        
        return answer
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        # ALWAYS try to use chunks as a fallback
        if chunks and len(chunks) > 0:
            try:
                st.warning("Falling back to direct chunk processing...")
                return generate_direct_response_with_chunks(query, chunks)
            except Exception as inner_e:
                st.error(f"Fallback also failed: {str(inner_e)}")
        
        return f"An error occurred: {str(e)}"