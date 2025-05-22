import streamlit as st
import os
import logging

# Impor fungsi inisialisasi dan GraphState dari modul bluguitar_core_logic
# Pastikan nama file 'bluguitar_core_logic.py' sudah benar dan ada di direktori yang sama
try:
    from bluguitar_core_logic import initialize_and_compile_graph_components, GraphState
except ImportError:
    st.error("Gagal mengimpor 'bluguitar_core_logic'. Pastikan file tersebut ada dan tidak ada error saat impor.")
    # Tambahkan logging di sini jika diperlukan untuk debugging path atau error impor
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.error("ImportError: Could not import from bluguitar_core_logic.", exc_info=True)
    st.stop() # Hentikan eksekusi jika modul inti tidak bisa diimpor


# --- Pengaturan Logging untuk Streamlit App (opsional, bisa ke file terpisah) ---
# Gunakan logger bernama agar tidak bentrok dengan logger modul lain jika ada
streamlit_logger = logging.getLogger("streamlit_app") 
streamlit_logger.setLevel(os.getenv("STREAMLIT_LOG_LEVEL", "INFO").upper())

# Hanya tambahkan handler jika belum ada (mencegah duplikasi jika Streamlit melakukan rerun)
if not streamlit_logger.handlers:
    log_file_path_streamlit = 'bluguitar_streamlit_app.log' # Nama file log khusus untuk Streamlit
    try:
        file_handler_streamlit = logging.FileHandler(log_file_path_streamlit)
        file_formatter_streamlit = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s')
        file_handler_streamlit.setFormatter(file_formatter_streamlit)
        streamlit_logger.addHandler(file_handler_streamlit)
    except Exception as e:
        # Jika ada masalah membuat file log, setidaknya tampilkan di konsol Streamlit (jika berjalan lokal)
        print(f"Peringatan: Gagal membuat file handler untuk logging Streamlit: {e}")
        # Dan pastikan ada handler dasar agar logger tidak error
        if not streamlit_logger.handlers: # Cek lagi untuk menghindari duplikasi jika print di atas juga gagal
            streamlit_logger.addHandler(logging.StreamHandler()) # Fallback ke console
# ------------------------------------------

# --- Fungsi Inisialisasi yang di-cache oleh Streamlit ---
@st.cache_resource # Cache resource agar tidak re-init setiap interaksi UI
def load_qna_application_and_tools():
    """
    Memanggil fungsi inisialisasi dari bluguitar_core_logic.py untuk mendapatkan
    graph yang sudah dikompilasi dan alat-alat lainnya.
    Fungsi ini akan di-cache oleh Streamlit.
    """
    streamlit_logger.info("Streamlit: Memulai inisialisasi QnA application dan tools...")
    try:
        # Panggil fungsi dari modul bluguitar_core_logic.py
        compiled_app, tavily_tool = initialize_and_compile_graph_components()
        if compiled_app:
            streamlit_logger.info("Streamlit: QnA application dan tools berhasil diinisialisasi dari core logic.")
            return compiled_app, tavily_tool
        else:
            streamlit_logger.error("Streamlit: Gagal mendapatkan compiled_app dari core logic. compiled_app adalah None.")
            st.error("Error: Komponen inti bot gagal diinisialisasi (compiled_app adalah None). Periksa log bluguitar_core_logic.log.")
            return None, None
    except Exception as e:
        # Tangkap error spesifik dari core logic jika ada
        st.error(f"Terjadi kesalahan fatal saat inisialisasi bot dari core logic: {e}")
        streamlit_logger.critical(f"Streamlit: Critical error during core logic initialization: {e}", exc_info=True)
        return None, None

# --- UI Streamlit ---
st.set_page_config(page_title="Bluguitar Q&A Bot", layout="wide", initial_sidebar_state="collapsed")
st.title("🎸 Bluguitar Q&A Bot")
st.caption("Tanyakan apa saja tentang produk, artis, atau informasi lain terkait Bluguitar!")

# Muat aplikasi dan alat (akan di-cache oleh Streamlit)
# st.session_state akan menyimpan compiled_qna_app agar tidak re-init setiap interaksi UI
# dan tersedia di seluruh sesi pengguna.
if 'app_initialized' not in st.session_state: # Flag untuk menandakan inisialisasi
    streamlit_logger.info("Streamlit: Melakukan panggilan pertama ke load_qna_application_and_tools().")
    compiled_qna_app_from_load, tavily_tool_from_load = load_qna_application_and_tools()
    if compiled_qna_app_from_load:
        st.session_state.compiled_qna_app = compiled_qna_app_from_load
        st.session_state.tavily_search_tool = tavily_tool_from_load 
        st.session_state.app_initialized = True # Tandai bahwa inisialisasi sudah dicoba
        # Tidak perlu st.success di sini karena cache_resource menanganinya, 
        # pesan error akan muncul jika load_qna_application_and_tools gagal.
        streamlit_logger.info("Streamlit App: Komponen inti berhasil diinisialisasi dan disimpan dalam session state.")
    else:
        st.session_state.compiled_qna_app = None 
        st.session_state.tavily_search_tool = None
        st.session_state.app_initialized = False # Tandai gagal
        # Pesan error sudah ditampilkan di dalam load_qna_application_and_tools()
        streamlit_logger.error("Streamlit App: Gagal menginisialisasi komponen inti untuk session state.")

# Inisialisasi riwayat chat di session_state jika belum ada
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Ada yang bisa saya bantu terkait Bluguitar hari ini?"}]

# Tampilkan pesan chat yang sudah ada dari session_state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input pertanyaan dari pengguna menggunakan st.chat_input
user_question = st.chat_input("Ketik pertanyaan Anda di sini...")

if user_question:
    # Tambahkan pertanyaan pengguna ke riwayat chat dan tampilkan di UI
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Dapatkan jawaban dari bot jika aplikasi sudah terinisialisasi dengan benar
    if st.session_state.get("compiled_qna_app"):
        app_to_run = st.session_state.compiled_qna_app
        tavily_tool_in_session = st.session_state.get("tavily_search_tool")

        # Logika untuk perintah /web jika Tavily tool tersedia
        if user_question.lower().startswith("/web ") and tavily_tool_in_session:
            query = user_question[5:].strip()
            if query:
                streamlit_logger.info(f"Streamlit: Performing Tavily web search for: {query}")
                with st.chat_message("assistant"): # Tampilkan jawaban di dalam chat message assistant
                    with st.spinner("Mencari di web menggunakan Tavily..."):
                        try:
                            results = tavily_tool_in_session.invoke({"query": query})
                            web_answer = "🌐 **Hasil Pencarian Web Tavily:**\n"
                            if isinstance(results, list) and results:
                                for i, item in enumerate(results):
                                    title = item.get('title', 'Tanpa Judul')
                                    url = item.get('url', '#')
                                    content_preview = item.get('content', '')[:150] # Ambil preview konten
                                    web_answer += f"\n{i+1}. **[{title}]({url})**\n   *\"{content_preview}...\"*\n"
                            elif isinstance(results, str): # Jika hasilnya hanya string tunggal
                                web_answer += results
                            else:
                                web_answer += "\n   Tidak ada hasil detail ditemukan atau format tidak dikenali."
                            st.markdown(web_answer)
                            st.session_state.messages.append({"role": "assistant", "content": web_answer})
                        except Exception as e:
                            st.error(f"Error saat melakukan pencarian web: {e}")
                            streamlit_logger.error(f"Streamlit: Tavily search error: {e}", exc_info=True)
                            error_message_for_chat = f"Maaf, terjadi error saat pencarian web: {str(e)[:100]}..."
                            st.markdown(error_message_for_chat) # Tampilkan error di chat
                            st.session_state.messages.append({"role": "assistant", "content": error_message_for_chat})
            else:
                st.warning("Mohon berikan query setelah perintah /web (contoh: /web sejarah gitar elektrik).")
        
        else: # Proses pertanyaan Bluguitar menggunakan graph utama
            with st.chat_message("assistant"): # Tampilkan jawaban di dalam chat message assistant
                with st.spinner("Memproses pertanyaan Anda tentang Bluguitar..."):
                    try:
                        streamlit_logger.info(f"Streamlit App: Processing Bluguitar question: {user_question}")
                        # Input untuk LangGraph adalah dict yang sesuai dengan field awal GraphState
                        graph_inputs = {"question": user_question} 
                        final_state = app_to_run.invoke(graph_inputs, {"recursion_limit": 10}) 
                        
                        bot_answer = final_state.get("answer", "Maaf, saya tidak bisa menghasilkan jawaban saat ini.")
                        
                        st.markdown(bot_answer)
                        st.session_state.messages.append({"role": "assistant", "content": bot_answer})

                        # (Opsional) Tampilkan sumber dokumen jika ada dan diinginkan (tetap tersembunyi dari UI utama)
                        if final_state.get("documents"):
                            streamlit_logger.info(f"Sources for '{user_question}': {[doc.metadata.get('title', 'Unknown') for doc in final_state['documents']]}")
                            # with st.expander("Lihat Sumber Dokumen (jika ada)"):
                            #     for i, doc in enumerate(final_state["documents"]):
                            #         metadata = doc.metadata
                            #         source_info = metadata.get("title") or os.path.basename(metadata.get("source_url", "Sumber Tidak Diketahui"))
                            #         st.write(f"Dok. {i+1}: {source_info}")
                        streamlit_logger.info("Streamlit App: Answer displayed.")

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses pertanyaan Anda: {e}")
                        streamlit_logger.error(f"Streamlit App: Error processing question: {e}", exc_info=True)
                        error_message_for_chat = f"Maaf, terjadi error: {str(e)[:100]}..."
                        st.markdown(error_message_for_chat) # Tampilkan error di chat
                        st.session_state.messages.append({"role": "assistant", "content": error_message_for_chat})
    else:
        st.error("Aplikasi bot belum siap atau gagal diinisialisasi. Mohon periksa pesan error di atas atau log aplikasi.")
        # Pesan ini akan muncul jika st.session_state.compiled_qna_app tidak ada atau None