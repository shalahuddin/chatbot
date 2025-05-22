import os
import logging
from dotenv import load_dotenv
from typing import TypedDict, List, Optional

# LangChain and LangGraph components
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults # Opsional
from langgraph.graph import END, StateGraph

# --- Pengaturan Logging untuk modul ini ---
core_logger = logging.getLogger(__name__)
core_logger.setLevel(os.getenv("CORE_LOG_LEVEL", "INFO").upper())

# Hanya tambahkan handler jika belum ada (mencegah duplikasi jika modul di-reload)
if not core_logger.handlers:
    log_file_path_core = 'bluguitar_core_logic.log'
    file_handler_core = logging.FileHandler(log_file_path_core)
    file_formatter_core = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s')
    file_handler_core.setFormatter(file_formatter_core)
    core_logger.addHandler(file_handler_core)

    # Opsional: Tambahkan StreamHandler untuk output ke konsol jika modul ini dijalankan langsung
    if __name__ == "__main__":
        console_handler_core_main = logging.StreamHandler()
        console_formatter_core_main = logging.Formatter('%(levelname)s: [%(name)s] %(message)s')
        console_handler_core_main.setLevel(logging.INFO) # Tampilkan INFO jika dijalankan langsung
        core_logger.addHandler(console_handler_core_main)
# ------------------------------------------

# --- Variabel Global dalam modul ini ---
# Akan diinisialisasi oleh initialize_and_compile_graph_components()
chroma_retriever = None
llm_model = None
bluguitar_qa_prompt_template = None
# tavily_search_tool dan compiled_qna_app akan dikembalikan oleh fungsi inisialisasi
# ------------------------------------

# --- Definisi State Graph ---
class GraphState(TypedDict): # Pastikan ini bisa diimpor oleh streamlit_app.py
    question: str
    documents: Optional[List[Document]]
    context: Optional[str]
    answer: Optional[str]

# --- Definisi Node Graph ---
# Node-node ini akan menggunakan variabel global yang di-set dalam modul ini
def retrieve_documents_node(state: GraphState) -> GraphState:
    global chroma_retriever # Mengakses global dari modul ini
    question = state["question"]
    core_logger.info(f"Node: retrieve_documents_node - Retrieving for: {question}")
    if not chroma_retriever:
        core_logger.error("Chroma Retriever not initialized!")
        return {**state, "documents": [], "context": "Error: Retriever not ready.", "answer": "System Error: Retriever component not initialized."}
    
    docs = chroma_retriever.invoke(question)
    context_str = "\n\n---\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant Bluguitar context found in the local database."
    core_logger.info(f"Retrieved {len(docs) if docs else 0} documents.")
    return {**state, "documents": docs, "context": context_str}

def generate_answer_node(state: GraphState) -> GraphState:
    global llm_model, bluguitar_qa_prompt_template # Mengakses global dari modul ini
    question = state["question"]
    context = state["context"]
    core_logger.info(f"Node: generate_answer_node - Generating answer for: {question}")

    if not llm_model or not bluguitar_qa_prompt_template:
        core_logger.error("LLM or QA Prompt not initialized!")
        return {**state, "answer": "System Error: LLM or Prompt Template not initialized."}

    prompt = bluguitar_qa_prompt_template.format(context=context, question=question)
    try:
        response = llm_model.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        core_logger.info(f"Answer generated: {answer[:100]}...") # Log sebagian jawaban
    except Exception as e:
        core_logger.error(f"LLM invocation error: {e}", exc_info=True)
        answer = "Sorry, an error occurred while generating the answer."
    return {**state, "answer": answer}

# --- Fungsi Inisialisasi Utama ---
def initialize_and_compile_graph_components():
    # Mendeklarasikan bahwa kita akan memodifikasi variabel global modul ini
    global chroma_retriever, llm_model, bluguitar_qa_prompt_template
    
    local_tavily_search_tool = None # Variabel lokal untuk Tavily tool

    load_dotenv() 
    core_logger.info("Core Logic: Environment variables loaded (if .env file exists).")
    core_logger.info("Core Logic: Initializing Bluguitar Q&A components...")

    # Konfigurasi Path dan Model (bisa diambil dari env vars untuk fleksibilitas)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.getenv("CHROMA_DB_PATH", os.path.join(base_dir, "chroma_db"))
    collection_name = os.getenv("CHROMA_COLLECTION", "bluguitar_docs")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash-latest")

    core_logger.info(f"Core Logic: ChromaDB persist directory set to: {persist_dir}")
    if not os.path.exists(persist_dir):
        core_logger.critical(f"Core Logic: ChromaDB directory NOT FOUND: {persist_dir}")
        raise FileNotFoundError(f"ChromaDB directory not found: {persist_dir}. Please ensure it exists or your CHROMA_DB_PATH is correct.")

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        core_logger.critical("Core Logic: GEMINI_API_KEY not found in environment variables.")
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file or environment.")
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        try:
            local_tavily_search_tool = TavilySearchResults(max_results=3)
            core_logger.info("Core Logic: Tavily Search tool initialized successfully.")
        except Exception as e:
            core_logger.warning(f"Core Logic: Failed to initialize Tavily Search tool: {e}. /web command might fail if used.")
    else:
        core_logger.info("Core Logic: TAVILY_API_KEY not found. Tavily Search tool will not be available.")

    embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_function, collection_name=collection_name)
    
    collection_count = vector_store._collection.count()
    core_logger.info(f"Core Logic: ChromaDB collection '{collection_name}' loaded with {collection_count} documents.")
    if collection_count == 0:
        core_logger.warning("Core Logic: Warning - ChromaDB collection is empty. Bot may not provide relevant answers from local data.")

    # Inisialisasi variabel global modul
    chroma_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm_model = ChatGoogleGenerativeAI(model=llm_model_name, temperature=0.1, google_api_key=gemini_api_key)
    
    bluguitar_prompt_text = """You are a helpful and knowledgeable AI assistant for Bluguitar.
Answer the user's question based ONLY on the provided Bluguitar Context.
If the user's question is very general (e.g., "what can it do?", "what is it?", "tell me about it") and the retrieved context is highly technical, very specific, or covers only a narrow aspect, try to provide a concise general overview if the context allows. If the context is insufficient for a general overview for such a general question, clearly state that you need a more specific question about Bluguitar products or features to provide a detailed answer from the documents.
If the provided context is not relevant to the question at all, or if no information is found for a specific question, clearly state that you could not find the information in the available Bluguitar documents.
Do not invent answers or use external knowledge. Keep your answer concise and directly related to the Bluguitar context.

Bluguitar Context:
{context}

Question: {question}

Helpful Answer:"""
    bluguitar_qa_prompt_template = PromptTemplate(input_variables=["context", "question"], template=bluguitar_prompt_text)

    # Membuat dan mengkompilasi graph
    workflow = StateGraph(GraphState)
    workflow.add_node("retriever", retrieve_documents_node)
    workflow.add_node("generator", generate_answer_node)
    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)
    
    compiled_app = workflow.compile()
    core_logger.info("Core Logic: Bluguitar Q&A LangGraph application compiled successfully.")
    
    # Kembalikan komponen yang dibutuhkan oleh pemanggil (misalnya, Streamlit)
    return compiled_app, local_tavily_search_tool

# --- Fungsi untuk menjalankan sesi interaktif terminal (jika file ini dijalankan langsung) ---
def run_terminal_interaction(compiled_app, tavily_tool):
    if not compiled_app:
        core_logger.critical("Core Logic: Application not compiled for terminal session. Exiting.")
        return

    print("\n🎸 Bluguitar Q&A Bot (Terminal Mode from Core Logic) 🎸")
    if tavily_tool:
        print("   Type '/web <query>' for web search.")
    print("   Type 'exit' or 'quit' to leave.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not user_input:
            continue

        if user_input.lower().startswith("/web ") and tavily_tool:
            query = user_input[5:].strip()
            if query:
                core_logger.info(f"Terminal: Performing Tavily web search for: {query}")
                try:
                    results = tavily_tool.invoke({"query": query})
                    print("\n🌐 Web Search Results:")
                    if isinstance(results, list) and results:
                        for i, item in enumerate(results):
                            print(f"  {i+1}. {item.get('title', 'N/A')} - {item.get('url', 'N/A')}")
                            # print(f"     Content: {item.get('content', '')[:100]}...") # Optional content preview
                    else:
                        print("  No detailed results found or format not recognized.")
                except Exception as e:
                    core_logger.error(f"Terminal: Tavily search error: {e}", exc_info=True)
                    print("  Sorry, an error occurred during web search.")
            else:
                print("  Please provide a query for web search (e.g., /web electric guitar history).")
        else: 
            core_logger.info(f"Terminal: Processing Bluguitar question: {user_input}")
            try:
                inputs = {"question": user_input}
                # Untuk LangGraph, recursion_limit bisa penting jika ada siklus atau banyak langkah
                # Untuk graph linear sederhana ini, mungkin tidak terlalu krusial tapi baik untuk ada
                final_state = compiled_app.invoke(inputs, {"recursion_limit": 10}) 
                
                print("\nBluguitar Bot:")
                print(final_state.get("answer", "No answer generated."))
                
                # Dokumen sumber tidak ditampilkan di konsol sesuai permintaan sebelumnya
                # Bisa diaktifkan jika perlu untuk debugging terminal
                # if final_state.get("documents"):
                #     core_logger.info(f"Source documents for terminal: {[doc.metadata for doc in final_state['documents']]}")


            except Exception as e:
                core_logger.error(f"Terminal: Error processing Bluguitar question: {e}", exc_info=True)
                print("  Sorry, an internal error occurred while processing your question.")

# --- Blok untuk menjalankan skrip ini secara langsung ---
if __name__ == "__main__":
    core_logger.info(f"{__file__} is being run directly.")
    # Tambahkan StreamHandler ke core_logger HANYA jika dijalankan langsung
    # Ini akan menampilkan log INFO dan ke atas ke konsol untuk sesi terminal
    # console_handler_main = logging.StreamHandler()
    # console_formatter_main = logging.Formatter('%(levelname)s: [%(name)s] %(message)s')
    # console_handler_main.setLevel(logging.INFO) 
    # core_logger.addHandler(console_handler_main)
    # core_logger.info("Console handler added for direct script execution.")


    try:
        # Panggil inisialisasi
        compiled_qna_app_main, tavily_tool_main = initialize_and_compile_graph_components()
        
        # Jalankan sesi interaktif terminal
        if compiled_qna_app_main:
            # Visualisasi ASCII Graph (opsional, hanya saat startup jika dijalankan langsung)
            try:
                print("\n--- Bluguitar Q&A Graph Structure (ASCII) ---")
                print(compiled_qna_app_main.get_graph().draw_ascii())
                print("--------------------------------------------\n")
            except Exception as e:
                core_logger.info(f"Could not print ASCII graph: {e}. Install 'grandalf' if needed.")
                pass
            run_terminal_interaction(compiled_qna_app_main, tavily_tool_main)
        else:
            core_logger.critical("Core Logic: Main QnA application could not be compiled. Terminal session cannot start.")
            
    except FileNotFoundError as e:
        core_logger.critical(f"Core Logic - CRITICAL ERROR (FileNotFound): {e}", exc_info=True)
        print(f"CRITICAL ERROR: {e}. Please ensure ChromaDB directory exists and paths are correct.")
    except ValueError as e:
        core_logger.critical(f"Core Logic - CRITICAL ERROR (ValueError): {e}", exc_info=True)
        print(f"CRITICAL ERROR: {e}. Please check your .env file for required API keys like GEMINI_API_KEY.")
    except Exception as e:
        core_logger.critical(f"Core Logic - An unexpected critical error occurred during script execution: {e}", exc_info=True)
        print(f"An unexpected critical error occurred. Please check logs ({log_file_path_core}).")