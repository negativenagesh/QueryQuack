from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import os
import re
from backend.model_utils import ensure_model_exists, MODELS_DIR

def estimate_tokens(text):
    """Roughly estimate the number of tokens in a text.
    
    For Phi-3, which uses a BPE tokenizer, we estimate about 3 characters per token.
    """
    return len(text) // 3

def generate_response(query, chunks):
    """Generate a response using Phi-3-mini with source attribution."""
    try:
        # Ensure model exists, download if not
        model_path = ensure_model_exists("phi-3-mini")
        if not model_path:
            return "Sorry, I couldn't generate a response. The model is not available."
            
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True
        )
        
        # Sort chunks by relevance (assuming they're already ordered by relevance)
        sorted_chunks = chunks
        
        # Format chunks into context
        context_parts = []
        for pdf, chunk, text in sorted_chunks:
            context_parts.append(f"{pdf} (Chunk {chunk}): {text}")
        
        context = "\n".join(context_parts)
        
        # Phi-3 uses instruction tuning format
        system_prompt = "You are a helpful AI assistant that provides accurate information based on the given context. Always cite your sources when providing information from documents."
        user_prompt = f"""Using ONLY the following context, answer this query: {query}

Context:
{context}

Answer concisely, citing the source PDF and chunk number. If the context doesn't contain relevant information to answer the query, respond with "No relevant information found."
"""
        
        # Create the complete prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Convert messages to model format
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Generate with Phi-3
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate the response
        with st.spinner("Generating response..."):
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=250,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up the response
        response = generated_text.strip()
        
        return response if response else "No relevant information found."
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"I encountered an error while generating a response: {str(e)}"