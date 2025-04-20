from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import os
import re
import torch
from backend.model_utils import ensure_model_exists, MODELS_DIR

def estimate_tokens(text):
    """Roughly estimate the number of tokens in a text.
    
    For TinyLlama, which uses a BPE tokenizer similar to Llama, we estimate about 4 characters per token.
    """
    return len(text) // 4

def generate_response(query, chunks):
    """Generate a response using TinyLlama with source attribution."""
    try:
        # Ensure model exists, download if not
        model_path = ensure_model_exists("tinyllama-1.1b-chat")
        if not model_path:
            return "Sorry, I couldn't generate a response. The model is not available."
            
        # Load model and tokenizer for CPU
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        # Set padding token to a value different from eos_token
        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token_id:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            local_files_only=True,
            # Using default float32 precision for CPU compatibility
            low_cpu_mem_usage=True      # Memory optimization for CPU
        )
        
        # Sort chunks by relevance (assuming they're already ordered by relevance)
        sorted_chunks = chunks
        
        # Format chunks into context, limiting context length for 8GB RAM constraints
        context_parts = []
        total_estimated_tokens = 0
        max_context_tokens = 1200  # Reduced for CPU-only usage
        
        for pdf, chunk, text in sorted_chunks:
            chunk_text = f"{pdf} (Chunk {chunk}): {text}"
            chunk_tokens = estimate_tokens(chunk_text)
            
            if total_estimated_tokens + chunk_tokens <= max_context_tokens:
                context_parts.append(chunk_text)
                total_estimated_tokens += chunk_tokens
            else:
                break  # Stop adding chunks if we exceed token limit
        
        context = "\n".join(context_parts)
        
        # TinyLlama follows Llama 2 chat format
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
        
        # Generate with TinyLlama - properly handle padding and attention mask
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            return_attention_mask=True  # Explicitly request attention mask
        )
        
        # Generate the response, using more memory-efficient settings
        with st.spinner("Generating response (this might take a while on CPU)..."):
            with torch.inference_mode():  # Use inference mode to save memory
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,  # Add attention mask
                    max_new_tokens=150,  # Further reduced for CPU usage
                    temperature=0.3,     # Keep temperature
                    top_p=0.9,           # Keep top_p
                    do_sample=True,      # Enable sampling to use temperature and top_p
                    pad_token_id=tokenizer.pad_token_id  # Use explicit pad token
                )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up the response
        response = generated_text.strip()
        
        return response if response else "No relevant information found."
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"I encountered an error while generating a response: {str(e)}"