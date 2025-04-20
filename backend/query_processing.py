import streamlit as st
import os
import numpy as np
from backend.model_utils import ensure_model_exists, MODELS_DIR

# Try importing advanced libraries with fallback
ADVANCED_QUERY_PROCESSING = False
try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ADVANCED_QUERY_PROCESSING = True
except (ImportError, RuntimeError):
    st.warning("Advanced query processing unavailable")

def rewrite_query(query):
    """Rewrite ambiguous queries using Phi-3-mini or return original."""
    if not ADVANCED_QUERY_PROCESSING:
        return query
        
    try:
        model_path = ensure_model_exists("phi-3-mini")
        if not model_path:
            return query
            
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True
        )
        
        # Create system and user messages for Phi-3
        system_message = "You are a helpful AI assistant that improves user queries to be more clear and precise. Keep the improved query concise."
        user_message = f"Rewrite this query to be clear and precise: {query}"
        
        # Format messages for the model
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Apply chat template to create the full prompt
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Encode the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate the response
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            temperature=0.1,  # Lower temperature for more deterministic output
            top_p=0.9,
            do_sample=False,  # No sampling for more consistent output
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Extract the generated response
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        rewritten_query = generated_text.strip()
            
        return rewritten_query if rewritten_query else query
    
    except Exception as e:
        st.warning(f"Query rewriting failed: {str(e)}. Using original query.")
        return query

def process_query(query, rewrite=True):
    """Convert query to embedding, optionally rewriting it."""
    try:
        if rewrite:
            query = rewrite_query(query)
            
        model_path = ensure_model_exists("all-MiniLM-L6-v2")
        if not model_path:
            return None, query
            
        model = SentenceTransformer(model_path)
        query_embedding = model.encode([query], convert_to_numpy=True)[0]
        
        # Convert numpy array to list
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
            
        return query_embedding, query
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None, query