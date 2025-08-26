from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_sample_data():
    
    documents = [
        "A RAG (Retrieval-Augmented Generation) olyan technika, amely összekapcsolja a információ-lekérést és a szöveggenerálást. A RAG rendszerek külső tudásbázisokat használnak a pontosabb válaszokért.",
        "A LangGraph egy könyvtár, amellyel agent-alapú alkalmazásokat lehet fejleszteni. A LangGraph segít modellezni komplex workflow-kat és állapot-gépeket.",
        "A ChromaDB egy nyílt forráskódú vektortár, beágyazások tárolására. Könnyen használható és hatékony vektorhasonlósági keresésre.",
        "Az agentic rendszerek autonóm döntéseket hoznak és részfeladatokra bontják a problémákat. Képesek külső eszközöket használni és összetett feladatokat megoldani.",
        "A Python egy népszerű programozási nyelv, amelyet gyakran használnak mesterséges intelligencia projektekhez. Egyszerű szintaxisa és gazdag könyvtára miatt népszerű."
    ]
    return documents

def split_into_chunks(documents, chunk_size=300, chunk_overlap=50):
    
    # Text splitter inicializálása
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  
    )
    
    # Összes dokumentum darabolása
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc)
        all_chunks.extend(chunks)
    
    return all_chunks


def process_documents(documents):
    
    return split_into_chunks(documents)
                        