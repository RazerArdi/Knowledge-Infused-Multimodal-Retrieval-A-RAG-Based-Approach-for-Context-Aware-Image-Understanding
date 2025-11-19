import streamlit as st
import os
import sys
import torch
import time
import pandas as pd
import numpy as np
import faiss
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, Blip2Processor, Blip2ForConditionalGeneration

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        h1 {font-family: 'Helvetica Neue', sans-serif; font-weight: 700; letter-spacing: -1px;}
        .stAlert {border-radius: 4px;}
        div[data-testid="stMetricValue"] {font-size: 1.4rem;}
        .img-caption {font-size: 0.8rem; color: #555;}
        /* Hide Streamlit footer */
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 1. BACKEND CONFIGURATION ---
# Kita mendefinisikan ulang Config di sini agar app.py bisa berdiri sendiri (Standalone)
class Config:
    # Path Management: Naik satu level dari folder 'User_Interface'
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    IMAGES_DIR = os.path.join(BASE_DIR, "Dataset", "Images")
    CAPTIONS_FILE = os.path.join(BASE_DIR, "Dataset", "captions.txt")
    
    # Menggunakan index yang sudah dibuat di Notebook
    INDEX_PATH = os.path.join(BASE_DIR, "flickr30k_large.index") 
    METADATA_PATH = os.path.join(BASE_DIR, "metadata_large.json")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model Definitions (Harus sama persis dengan Notebook)
    RETRIEVAL_MODEL = "openai/clip-vit-large-patch14"
    CAPTION_MODEL = "Salesforce/blip2-opt-2.7b"
    LLM_API = "http://localhost:11434/api/generate"
    LLM_MODEL = "llama3"

# --- 2. CORE CLASSES (CACHED) ---
# Menggunakan @st.cache_resource agar model tidak di-load ulang setiap interaksi

@st.cache_resource(show_spinner=False)
def load_retrieval_system():
    """Memuat CLIP Model dan FAISS Index."""
    
    # A. Load Encoder
    print(f"Loading Retrieval Model: {Config.RETRIEVAL_MODEL}...")
    processor = CLIPProcessor.from_pretrained(Config.RETRIEVAL_MODEL)
    model = CLIPModel.from_pretrained(Config.RETRIEVAL_MODEL).to(Config.DEVICE)
    model.eval()
    
    # B. Load Vector Database
    print(f"Loading FAISS Index from: {Config.INDEX_PATH}...")
    if not os.path.exists(Config.INDEX_PATH):
        st.error(f"Index file not found at {Config.INDEX_PATH}. Please run the Notebook first.")
        return None, None, None
        
    index = faiss.read_index(Config.INDEX_PATH)
    
    # C. Load Metadata
    import json
    with open(Config.METADATA_PATH, 'r') as f:
        metadata = json.load(f)
        
    return processor, model, (index, metadata)

@st.cache_resource(show_spinner=False)
def load_generative_system():
    """Memuat BLIP-2 Model."""
    print(f"Loading Generative Model: {Config.CAPTION_MODEL}...")
    processor = Blip2Processor.from_pretrained(Config.CAPTION_MODEL)
    # Load dengan float16 untuk efisiensi memori
    dtype = torch.float16 if Config.DEVICE == "cuda" else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(
        Config.CAPTION_MODEL, torch_dtype=dtype
    ).to(Config.DEVICE)
    return processor, model

# --- 3. HELPER FUNCTIONS ---

def perform_retrieval(query_text, clip_processor, clip_model, vector_db, k=5):
    index, metadata = vector_db
    
    # Embed Text
    text_with_prompt = [f"A photo of {query_text}"]
    inputs = clip_processor(text=text_with_prompt, return_tensors="pt", padding=True).to(Config.DEVICE)
    
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    
    # Search FAISS
    q_vec = features.cpu().numpy().astype('float32')
    distances, indices = index.search(q_vec, k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1:
            results.append({
                "filename": metadata[idx],
                "score": float(dist),
                "path": os.path.join(Config.IMAGES_DIR, metadata[idx])
            })
    return results

def generate_context(images_data, blip_processor, blip_model):
    contexts = []
    for item in images_data:
        try:
            raw_image = Image.open(item['path']).convert('RGB')
            dtype = torch.float16 if Config.DEVICE == "cuda" else torch.float32
            inputs = blip_processor(images=raw_image, return_tensors="pt").to(Config.DEVICE, dtype)
            
            out = blip_model.generate(**inputs, max_new_tokens=50)
            caption = blip_processor.decode(out[0], skip_special_tokens=True).strip()
            contexts.append(caption)
        except Exception as e:
            contexts.append(f"[Error analyzing image: {str(e)}]")
    return contexts

def query_llm(context_list, user_query):
    context_str = "\n".join([f"- Image Evidence {i+1}: {txt}" for i, txt in enumerate(context_list)])
    
    prompt = f"""
    [INST]
    You are a helpful visual AI assistant. You have been provided with descriptions of images retrieved from a database that match the user's query.
    
    Visual Evidence:
    {context_str}
    
    User Question: "{user_query}"
    
    Based strictly on the visual evidence provided above, answer the user's question. Provide a cohesive narrative. If the evidence is not sufficient, state that clearly.
    [/INST]
    """
    
    payload = {
        "model": Config.LLM_MODEL, 
        "prompt": prompt, 
        "stream": False
    }
    
    try:
        response = requests.post(Config.LLM_API, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "Error: Empty response from Ollama.")
        else:
            return f"Error: LLM Service returned status {response.status_code}"
    except Exception as e:
        return f"Error connecting to LLM Service: {e}"

# --- 4. MAIN UI LAYOUT ---

def main():
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        top_k = st.slider("Retrieval Count (K)", min_value=1, max_value=10, value=5)
        
        st.divider()
        st.caption("System Status")
        
        # Status Indicators
        with st.status("Checking Resources...", expanded=True) as status:
            st.write("⚡ Device:", Config.DEVICE.upper())
            
            # Load Retrieval System
            clip_proc, clip_model, vector_db = load_retrieval_system()
            if vector_db:
                st.success("✅ Retrieval Engine Ready")
            else:
                st.error("❌ Retrieval Engine Failed")
                status.update(state="error")
                st.stop()

            # Load Generative System
            blip_proc, blip_model = load_generative_system()
            st.success("✅ Generative Engine Ready")
            
            status.update(label="All Systems Operational", state="complete", expanded=False)
        
        st.divider()
        st.info(
            "**About:**\n"
            "This system utilizes RAG (Retrieval Augmented Generation) to answer questions based on visual evidence from the Flickr30k dataset."
        )

    # Main Content
    st.title("Multimodal RAG System")
    st.markdown("##### A RAG-Based Approach to Image Retrieval and Context-Aware Generation")
    
    # Input Area
    query = st.text_input("Enter your visual query:", placeholder="e.g., Two men playing guitar in the park...")
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    if run_btn and query:
        if not query.strip():
            st.warning("Please enter a valid query.")
            return

        # --- PIPELINE EXECUTION ---
        start_time = time.time()
        
        # 1. Retrieval Phase
        with st.spinner("🔍 Searching Vector Database..."):
            retrieved_items = perform_retrieval(query, clip_proc, clip_model, vector_db, k=top_k)
            retrieval_time = time.time() - start_time

        # Layout: Columns for Results
        col_left, col_right = st.columns([1.2, 1])

        # 2. Display Retrieval Results (Left Column)
        with col_left:
            st.subheader(f"Evidence (Top-{top_k})")
            st.caption(f"Retrieval Latency: {retrieval_time:.4f}s")
            
            # Display as a clean grid or list
            tabs = st.tabs([f"Rank {i+1}" for i in range(len(retrieved_items))])
            
            visual_contexts = [] # Store captions for LLM
            
            for i, (tab, item) in enumerate(zip(tabs, retrieved_items)):
                with tab:
                    try:
                        img = Image.open(item['path'])
                        st.image(img, use_column_width=True)
                        
                        # Metadata Container
                        with st.expander("See Metadata", expanded=True):
                            st.code(f"Filename: {item['filename']}", language="text")
                            st.progress(item['score'], text=f"Similarity Score: {item['score']:.4f}")
                    except Exception as e:
                        st.error(f"Image load error: {e}")

        # 3. Generative Phase (Right Column)
        with col_right:
            st.subheader("Context-Aware Generation")
            
            # BLIP-2 Processing
            with st.status("Analyzing Visual Context...", expanded=True) as gen_status:
                st.write("👁️ Generating captions with BLIP-2...")
                visual_contexts = generate_context(retrieved_items, blip_proc, blip_model)
                
                st.write("🧠 Reasoning with Llama-3...")
                final_answer = query_llm(visual_contexts, query)
                
                gen_status.update(label="Generation Complete", state="complete", expanded=False)
            
            # Display Answer
            st.markdown("### AI Response")
            st.success(final_answer)
            
            # Display Generated Contexts (Transparency)
            with st.expander("View Generated Visual Contexts (BLIP-2 Output)"):
                for idx, ctx in enumerate(visual_contexts):
                    st.text(f"Img {idx+1}: {ctx}")

        # Total Time Footer
        total_time = time.time() - start_time
        st.divider()
        st.caption(f"Total Pipeline Latency: {total_time:.2f} seconds | Powered by CLIP, FAISS, BLIP-2, & Llama-3")

if __name__ == "__main__":
    main()