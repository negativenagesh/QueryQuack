import os
import streamlit as st
import shutil

# Define path to models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def ensure_model_exists(model_name):
    """
    Check if model exists locally, download if not.
    Returns the path to the model.
    """
    model_path = os.path.join(MODELS_DIR, model_name)
    
    # Create models directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Different check files for different model types
    if model_name == "phi-3-mini":
        check_file = os.path.join(model_path, "tokenizer_config.json")
    else:
        check_file = os.path.join(model_path, "config.json")
        
    if not os.path.exists(check_file):
        st.info(f"Downloading model: {model_name}. This may take a few minutes...")
        try:
            if model_name == "phi-3-mini":
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    # Download to a temporary directory first
                    temp_path = os.path.join(MODELS_DIR, f"{model_name}_temp")
                    os.makedirs(temp_path, exist_ok=True)
                    
                    # Download both model and tokenizer
                    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
                    model = AutoModelForCausalLM.from_pretrained(
                        "microsoft/phi-3-mini-4k-instruct", 
                        torch_dtype="auto", 
                        device_map="auto"
                    )
                    
                    # Save to the temporary location
                    tokenizer.save_pretrained(temp_path)
                    model.save_pretrained(temp_path)
                    
                    # If successful, move to the actual model path
                    if os.path.exists(model_path):
                        shutil.rmtree(model_path)
                    shutil.move(temp_path, model_path)
                except Exception as e:
                    st.error(f"Error downloading {model_name}: {str(e)}")
                    return None
            elif model_name == "all-MiniLM-L6-v2":
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                    model.save(model_path)
                except Exception as e:
                    st.error(f"Error downloading {model_name}: {str(e)}")
                    return None
            elif model_name == "ms-marco-MiniLM-L-6-v2":
                try:
                    from sentence_transformers import CrossEncoder
                    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                    model.save(model_path)
                except Exception as e:
                    st.error(f"Error downloading {model_name}: {str(e)}")
                    return None
            else:
                st.error(f"Unknown model: {model_name}")
                return None
        except Exception as e:
            st.error(f"Error downloading {model_name}: {str(e)}")
            return None
        st.success(f"Model {model_name} downloaded successfully!")
    return model_path