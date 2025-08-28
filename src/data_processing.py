from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import os

def load_sample_data():
    
    documents = [
        "A RAG (Retrieval-Augmented Generation) olyan technika, amely összekapcsolja a információ-lekérést és a szöveggenerálást. A RAG rendszerek külső tudásbázisokat használnak a pontosabb válaszokért.",
        "A LangGraph egy könyvtár, amellyel agent-alapú alkalmazásokat lehet fejleszteni. A LangGraph segít modellezni komplex workflow-kat és állapot-gépeket.",
        "A ChromaDB egy nyílt forráskódú vektortár, beágyazások tárolására. Könnyen használható és hatékony vektorhasonlósági keresésre.",
        "Az agentic rendszerek autonóm döntéseket hoznak és részfeladatokra bontják a problémákat. Képesek külső eszközöket használni és összetett feladatokat megoldani.",
        "A Python egy népszerű programozási nyelv, amelyet gyakran használnak mesterséges intelligencia projektekhez. Egyszerű szintaxisa és gazdag könyvtára miatt népszerű."
    ]
    return documents

def load_data(data_path="./data"):
    documents = []

    if not os.path.exists(data_path):
        print(f"A megadott '{data_path}' mappa nem létezik. Mintaadatok használata...")
        return load_sample_data()  # Itt HIÁNYZOTT a () zárójelpár a függvényhíváshoz!

    print(f"Fájlok betöltése innen: {data_path}")

    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)
        print(f"Feldolgozás: {filepath}")  # Debug info

        try:
            if filename.endswith(".pdf"):
                # Átadjuk a konkrét FÁJL elérési utat, nem a mappáét
                text = extract_text_from_pdf(filepath)
                if text:  # Csak akkor adjuk hozzá, ha sikerült szöveget kinyerni
                    documents.append(text)
                    print(f"A PDF fájl betöltve: {filename}")
                else:
                    print(f"A PDF fájl feldolgozása sikertelen: {filename}")
            elif filename.endswith(".txt"):
                with open(filepath, "r", encoding='utf-8') as f:
                    text = f.read()
                documents.append(text)
                print(f"TXT dokumentum betöltve: {filename}")
            else:
                print(f"A(z) {filename} kiterjesztése nem támogatott, kihagyva.")

        except Exception as e:
            print(f"Hiba történt a(z) {filename} feldolgozása során: {e}")
            # Itt nem feltétlenül érdemes rögtön visszatérni a mintaadatokkal,
            # hanem csak jelzünk és megyünk tovább a következő fájlra.
            # return load_sample_data()

    if len(documents) == 0:
        print("Nem sikerült betölteni egyetlen dokumentumot sem a mappából. Mintaadatok használata...")
        return load_sample_data()

    print(f"Összesen {len(documents)} dokumentum betöltve")
    return documents

def extract_text_from_pdf(pdf_path):
    """Kinyeri a szöveget egy PDF fájlból."""
    text = ""
    try:
        # A PdfReader a pypdf csomag része
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            # extract_text() metódust hívunk, nem extract_text
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Hiba a PDF olvasása során ({pdf_path}): {e}")
        return None  # Explicit None return on error
    return text  # Return the extracted text

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
                        