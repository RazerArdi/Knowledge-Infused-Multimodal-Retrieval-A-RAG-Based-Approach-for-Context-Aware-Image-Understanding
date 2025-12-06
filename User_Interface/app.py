import streamlit as st
import os
import sys
import torch
import time
import pandas as pd
import numpy as np
import faiss
import requests
import gc
import json
from PIL import Image
from collections import defaultdict
from difflib import SequenceMatcher
from transformers import CLIPProcessor, CLIPModel, Blip2Processor, Blip2ForConditionalGeneration

# --- 0. CRITICAL: ENVIRONMENT SETUP ---
os.environ['HF_HOME'] = "/mnt/d/huggingface_cache" 

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MEMORY MANAGEMENT ---
def clear_memory():
    """
    Forces garbage collection and clears CUDA cache to prevent Out-Of-Memory (OOM) errors.
    This function should be called before and after heavy inference tasks.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        h1 {font-family: 'Helvetica Neue', sans-serif; font-weight: 700; letter-spacing: -1px;}
        .stAlert {border-radius: 4px;}
        div[data-testid="stMetricValue"] {font-size: 1.4rem;}
        .img-caption {font-size: 0.8rem; color: #555;}
        .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px;}
        .gt-caption {font-size: 0.85rem; color: #2e7d32; border-left: 2px solid #2e7d32; padding-left: 10px; margin-bottom: 5px;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 1. BACKEND CONFIGURATION ---
class Config:
    """
    Central configuration class for file paths, model selection, and API endpoints.
    """
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    IMAGES_DIR = os.path.join(BASE_DIR, "Dataset", "Images")
    CAPTIONS_FILE = os.path.join(BASE_DIR, "Dataset", "captions.txt")
    
    INDEX_PATH = os.path.join(BASE_DIR, "Notebook", "flickr30k_large.index") 
    METADATA_PATH = os.path.join(BASE_DIR, "Notebook", "metadata_large.json")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    RETRIEVAL_MODEL = "openai/clip-vit-large-patch14"
    CAPTION_MODEL = "Salesforce/blip2-opt-2.7b"
    
    LLM_API_LOCAL = "http://localhost:11434/api/generate"
    LLM_API_WIN = "http://172.29.240.1:11434/api/generate"
    
    LLM_API = LLM_API_LOCAL 
    LLM_MODEL = "llama3" 

# --- 2. CORE CLASSES (SMART LOADER) ---

@st.cache_resource(show_spinner=False)
def load_captions_dataset():
    """
    Loads ground truth captions from the CSV/Text dataset into a dictionary.
    
    Returns:
        defaultdict: Mapping of image filenames to a list of ground truth captions.
    """
    print(f"📂 Loading Captions from: {Config.CAPTIONS_FILE}")
    captions_dict = defaultdict(list)
    
    if not os.path.exists(Config.CAPTIONS_FILE):
        st.warning(f"⚠️ Captions file not found at {Config.CAPTIONS_FILE}")
        return captions_dict

    try:
        with open(Config.CAPTIONS_FILE, 'r', encoding='utf-8') as f:
            next(f) # Skip header
            for line in f:
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    img_name, cap = parts
                    clean_cap = cap.strip().strip('"')
                    captions_dict[img_name].append(clean_cap)
        return captions_dict
    except Exception as e:
        st.error(f"Error reading captions file: {e}")
        return captions_dict

@st.cache_resource(show_spinner=False)
def load_retrieval_system():
    """
    Initializes the CLIP model and loads the FAISS index for vector retrieval.
    Includes logic to auto-detect index dimension and switch CLIP models accordingly.
    
    Returns:
        tuple: (CLIPProcessor, CLIPModel, (faiss_index, metadata))
    """
    clear_memory()
    
    if not os.path.exists(Config.INDEX_PATH):
        st.error(f"⚠️ Index file missing at `{Config.INDEX_PATH}`.")
        return None, None, None
        
    try:
        index = faiss.read_index(Config.INDEX_PATH)
        disk_dim = index.d
        print(f"📏 Detected Index Dimension: {disk_dim}")
        
        if disk_dim == 512:
            print("Switching to CLIP-BASE (512) to match index.")
            Config.RETRIEVAL_MODEL = "openai/clip-vit-base-patch32"
        elif disk_dim == 768:
            print("Switching to CLIP-LARGE (768) to match index.")
            Config.RETRIEVAL_MODEL = "openai/clip-vit-large-patch14"
        else:
            st.warning(f"Unknown index dimension: {disk_dim}. Keeping default.")
            
    except Exception as e:
        st.error(f"Error reading index: {e}")
        return None, None, None
    
    if not os.path.exists(Config.METADATA_PATH):
         st.error(f"Metadata json missing at {Config.METADATA_PATH}")
         return None, None, None
         
    with open(Config.METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    print(f"Allocating Encoder: {Config.RETRIEVAL_MODEL}...")
    processor = CLIPProcessor.from_pretrained(Config.RETRIEVAL_MODEL)
    model = CLIPModel.from_pretrained(Config.RETRIEVAL_MODEL).to(Config.DEVICE)
    model.eval()
        
    return processor, model, (index, metadata)

@st.cache_resource(show_spinner=False)
def load_generative_system():
    """
    Initializes the BLIP-2 model for image captioning using float16 precision.
    
    Returns:
        tuple: (Blip2Processor, Blip2ForConditionalGeneration)
    """
    clear_memory()
    print(f"Loading Gen Model: {Config.CAPTION_MODEL}...")
    
    dtype = torch.float16 if Config.DEVICE == "cuda" else torch.float32
    
    processor = Blip2Processor.from_pretrained(Config.CAPTION_MODEL)
    model = Blip2ForConditionalGeneration.from_pretrained(
        Config.CAPTION_MODEL, 
        torch_dtype=dtype
    ).to(Config.DEVICE)
    
    return processor, model

# --- 3. HELPER FUNCTIONS ---

def perform_retrieval(query_text, clip_processor, clip_model, vector_db, k=5):
    """
    Executes semantic search using CLIP embeddings and FAISS.
    Performs L2 normalization on query vectors to match IndexFlatIP (Cosine Similarity).
    
    Args:
        query_text (str): User query.
        k (int): Number of top results.
        
    Returns:
        list: Retrieval results.
    """
    index, metadata = vector_db
    
    text_with_prompt = [f"A photo of {query_text}"]
    inputs = clip_processor(text=text_with_prompt, return_tensors="pt", padding=True).to(Config.DEVICE)
    
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    
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
            
    del inputs, features, q_vec
    clear_memory()
    return results

def generate_context(images_data, blip_processor, blip_model, user_query_text=""):
    """
    Generates visual descriptions using BLIP-2.
    Uses 'Pure Visual' mode (no text prompt) to prevent hallucination.
    Reduced 'num_return_sequences' to prevent CUDA OOM errors.
    
    Args:
        images_data (list): List of image data.
    
    Returns:
        list: Generated context strings.
    """
    contexts = []
    dtype = torch.float16 if Config.DEVICE == "cuda" else torch.float32
    
    for item in images_data:
        try:
            img_path = item['path']
            if not os.path.isabs(img_path):
                img_path = os.path.abspath(img_path)
                
            raw_image = Image.open(img_path).convert('RGB')
            
            # [1. PURE VISUAL INPUT]
            inputs = blip_processor(images=raw_image, return_tensors="pt").to(Config.DEVICE, dtype)
            
            # [2. SAFE SETTINGS FOR VRAM]
            with torch.no_grad():
                out = blip_model.generate(
                    **inputs, 
                    max_new_tokens=80,
                    min_length=15,
                    do_sample=True,          
                    top_p=0.90,              
                    temperature=0.6,         
                    repetition_penalty=1.2,
                    num_return_sequences=5  # [FIX] Reduced from 15 to 5 to prevent OOM
                )
            
            all_candidates = blip_processor.batch_decode(out, skip_special_tokens=True)
            
            # [3. DEDUPLICATION]
            unique_captions = []
            for cap in all_candidates:
                clean_cap = cap.strip()
                if not clean_cap: continue
                
                is_duplicate = False
                for existing in unique_captions:
                    similarity = SequenceMatcher(None, clean_cap.lower(), existing.lower()).ratio()
                    if similarity > 0.8: 
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_captions.append(clean_cap)
            
            if not unique_captions: unique_captions = ["Image content unclear."]
            
            main_paragraph = unique_captions[0]
            remaining_points = unique_captions[1:] 
            
            if remaining_points:
                bullet_points = "\n".join([f"• {cap}" for cap in remaining_points])
                final_text = f"{main_paragraph}\n\n{bullet_points}"
            else:
                final_text = main_paragraph
                
            contexts.append(final_text)
            
            # [4. CRITICAL MEMORY CLEANUP]
            del inputs, out, raw_image
            clear_memory()
            
        except Exception as e:
            contexts.append(f"[Error: {str(e)}]")
            clear_memory()
            
    return contexts

def check_ollama_status():
    """
    Checks if Ollama API is reachable.
    
    Returns:
        tuple: (Status, URL)
    """
    urls = [Config.LLM_API_LOCAL, Config.LLM_API_WIN]
    for url in urls:
        try:
            test_url = url.replace("/generate", "/tags")
            resp = requests.get(test_url, timeout=1)
            if resp.status_code == 200:
                return True, url
        except:
            continue
    return False, None

def query_llm(context_list, user_query, api_url):
    """
    Audits evidence using Llama-3 iteratively (one image at a time) to ensure no rank is skipped.
    Integrates Ground Truth and AI Vision for the audit.
    
    Args:
        context_list (list): Combined context strings (BLIP + GT).
        user_query (str): User's query.
        
    Returns:
        str: Full audit report.
    """
    full_report = ""
    
    progress_bar = st.progress(0, text="🕵️ Llama-3 is auditing evidence...")
    
    for i, context_item in enumerate(context_list):
        rank = i + 1
        progress_bar.progress((i + 1) / len(context_list), text=f"Auditing Rank {rank}/{len(context_list)}...")

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Strict Data Auditor.
Your task is to verify if ONE specific image matches the User Query based on provided evidence.

GOLDEN RULES:
1. **TRUST DATASET FACTS:** The [Dataset Facts] section contains 100% accurate human labels.
2. **PRIORITY:** If [Dataset Facts] confirm the query details (e.g. "young boy", "market"), it is a MATCH.
3. **IGNORE BLIP IF WRONG:** If [AI Vision] contradicts [Dataset Facts], ignore [AI Vision].

OUTPUT FORMAT:
**RANK {rank} Analysis:**
- **Verdict:** [MATCH / PARTIAL / NO MATCH]
- **Audit:**
  - Query asked for: "[Keyword]"
  - Dataset Facts say: "[Quote relevant fact]"
  - AI Vision says: "[Quote relevant vision]"
- **Conclusion:** [One sentence final decision]

CONTEXT DATA:
{context_item}

<|eot_id|><|start_header_id|>user<|end_header_id|>
Query: "{user_query}"
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        payload = {
            "model": Config.LLM_MODEL, 
            "prompt": prompt, 
            "stream": False,
            "options": {
                "num_predict": 512,
                "temperature": 0.0,
                "top_p": 1.0
            }
        }

        def send_request():
            return requests.post(api_url, json=payload, timeout=30)

        try:
            response = send_request()
            
            if response.status_code == 500:
                time.sleep(1)
                response = send_request()
                
            if response.status_code == 200:
                answer_fragment = response.json().get("response", "").strip()
                full_report += answer_fragment + "\n\n" + ("-"*40) + "\n\n"
            
            elif response.status_code == 404:
                full_report += f"**RANK {rank}:** Error - Model not found.\n\n"
            else:
                full_report += f"**RANK {rank}:** Error - API Failure.\n\n"
                
        except Exception as e:
            full_report += f"**RANK {rank}:** Connection Error: {e}\n\n"
    
    progress_bar.empty()
    return full_report

# --- 4. MAIN UI LAYOUT ---

def main():
    with st.sidebar:
        st.header("Configuration")
        top_k = st.slider("Retrieval Count (K)", min_value=1, max_value=10, value=3)
        st.markdown("---")
        
        with st.expander("📐 Vector Search Method", expanded=True):
            st.markdown("""
            **Metric:** Cosine Similarity
            
            **Implementation:**
            $$
            \\text{Sim}(A, B) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}
            $$
            """)
        
        st.markdown("---")
        if st.button("🧹 Flush RAM/VRAM"):
            clear_memory()
            st.toast("Memory cleared!", icon="🧹")
            
        st.divider()
        
        with st.status("System Status", expanded=True) as status:
            gt_captions = load_captions_dataset()
            st.success(f"✅ Captions Loaded ({len(gt_captions)} items)")

            clip_proc, clip_model, vector_db = load_retrieval_system()
            if vector_db: st.success(f"✅ Retrieval Ready")
            else: st.stop()

            blip_proc, blip_model = load_generative_system()
            st.success("✅ Generative Ready")
            
            ollama_ok, working_url = check_ollama_status()
            if ollama_ok:
                st.success(f"✅ Llama-3 Connected")
                Config.LLM_API = working_url 
            else:
                st.warning("⚠️ Ollama Offline")
            
            status.update(label="System Operational", state="complete", expanded=False)
        
        st.info("**About:** Multimodal RAG System using Flickr30k.")

    st.title("Multimodal RAG System")
    st.markdown("##### A RAG-Based Approach to Image Retrieval and Context-Aware Generation")
    
    query = st.text_input("Enter your visual query:", placeholder="e.g., Two men playing guitar in the park...")
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    if run_btn and query:
        if not query.strip():
            st.warning("Please enter a valid query.")
            return

        clear_memory()
        start_time = time.time()
        
        # 1. RETRIEVAL PHASE
        with st.spinner("🔍 Searching Vector Database..."):
            ret_start = time.time()
            results = perform_retrieval(query, clip_proc, clip_model, vector_db, k=top_k)

        # Metrics
        scores = [r['score'] for r in results]
        st.markdown("### 📊 Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retrieval Latency", f"{(time.time()-start_time)*1000:.1f} ms")
        m2.metric("Top-1 Score", f"{scores[0]:.4f}")
        m3.metric("Avg Score", f"{np.mean(scores):.4f}")
        m4.metric("Items", f"{len(results)}")
        st.divider()

        col_left, col_right = st.columns([1.2, 1])

        # 2. EVIDENCE DISPLAY
        with col_left:
            st.subheader(f"Evidence (Top-{top_k})")
            tabs = st.tabs([f"Rank {i+1}" for i in range(len(results))])
            
            for i, (tab, item) in enumerate(zip(tabs, results)):
                with tab:
                    try:
                        img = Image.open(item['path'])
                        st.image(img, use_container_width=True)
                        
                        fname = item['filename']
                        st.markdown("**📂 Ground Truth Captions:**")
                        if fname in gt_captions:
                            with st.container(height=150, border=True):
                                for cap in gt_captions[fname]:
                                    st.markdown(f"<div class='gt-caption'>• {cap}</div>", unsafe_allow_html=True)
                        else:
                            st.caption("No ground truth found.")

                        with st.expander("See Metadata"):
                            st.code(f"File: {fname}\nScore: {item['score']:.4f}")
                    except Exception as e:
                        st.error(f"Image error: {e}")

        # 3. GENERATIVE PHASE
        with col_right:
            st.subheader("Context-Aware Generation")
            
            with st.status("Processing...", expanded=True) as gen_status:
                st.write("👁️ BLIP-2: Analyzing Visuals...")
                visual_contexts = generate_context(results, blip_proc, blip_model)
                
                # DATA FUSION
                st.write("📂 Data Fusion: Merging Vision + Dataset Facts...")
                rich_context_for_llm = []
                
                for i, blip_text in enumerate(visual_contexts):
                    fname = results[i]['filename']
                    
                    gt_text = "(No verified data available)"
                    if fname in gt_captions:
                        gt_list = gt_captions[fname][:10]
                        gt_text = "\n".join([f"- {c}" for c in gt_list])
                    
                    combined_entry = (
                        f"=== IMAGE #{i+1} (Rank {i+1}) ===\n"
                        f"[AI Vision / BLIP-2 Output]:\n{blip_text}\n\n"
                        f"[Dataset Facts / Ground Truth]:\n{gt_text}\n"
                        f"====================================="
                    )
                    rich_context_for_llm.append(combined_entry)

                st.write("🧠 Llama-3: Auditing Evidence...")
                if ollama_ok:
                    gen_start = time.time()
                    final_answer = query_llm(rich_context_for_llm, query, Config.LLM_API)
                    gen_latency = time.time() - gen_start
                else:
                    final_answer = "⚠️ Ollama error."
                    gen_latency = 0
                
                gen_status.update(label="Done", state="complete", expanded=False)
            
            # MINIMIZE UI
            with st.expander("🤖 View AI Response (Llama-3 Audit)", expanded=True):
                if "Error" in final_answer:
                    st.error(final_answer)
                else:
                    st.markdown(final_answer)
                    st.caption(f"Latency: {gen_latency:.2f}s")
            
            with st.expander("👁️ View Generated Visual Contexts (BLIP-2 Output)", expanded=False):
                for idx, ctx in enumerate(visual_contexts):
                    st.markdown(f"**Img {idx+1}:**")
                    st.text(ctx)
                    st.divider()

        st.divider()
        st.caption(f"Total Time: {time.time() - start_time:.2f}s")
        clear_memory()

if __name__ == "__main__":
    main()