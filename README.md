# Agentic RAG Chatbot
Ez a projekt egy egyszerű, modulárisan felépített agentic RAG chatbot prototípus, amely LangGraph segítségével workflow-ként kezeli a kérdések feldolgozását. A rendszer képes eldönteni, hogy egy kérdésre közvetlen választ adjon, vagy külső tudásbázisból (vektortárból) keressen információt.

# Főbb technológiák

- LangGraph – állapotgép-alapú workflow modellezés
- LangChain – szövegfeldolgozás és chunkolás
- ChromaDB – vektortár és keresés
- SentenceTransformers – beágyazás generálás

# Agent működése

Az agent a kérdés alapján dönt:
- `search_knowledge`: ha a kérdés információt igényel, a vektortárból keres
- `direct_response`: ha nem szükséges keresés, sablonos választ ad

A válaszgenerálás jelenleg szimulált, de a rendszer úgy van kialakítva, hogy könnyen lecserélhető legyen egy LLM API-ra vagy lokálisan futtatott modellre.


# Futtatás

1. Telepítsd a függőségeket:
   ```bash
   pip install -r requirements.txt

2.  Nyisd meg a demo.ipynb notebookot, és futtasd lépésenként.
    - A notebook bemutatja:
    - Adatfeldolgozást
    - Vektortár feltöltését
    - Agent működését
    - LangGraph workflow-t
    - Tesztkérdésekre adott válaszokat
