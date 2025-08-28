class SimpleAgent:
    def __init__(self,vector_store):
        self.vector_store = vector_store

    def decide_action(self,question):

        keywords = ["neural network", "embeddings", "softmax", "regularization", "nvidia"]
        
        for keyword in keywords:
            if keyword in question.lower():
                print("Vektor")
                return "search_knowledge"
        print("nem vektor")
        return "direct_response"
    
    def process_question(self,question):

        action = self.decide_action(question)

        print(f"Az agent döntése {action}")
        
        if action == "search_knowledge":
            print("Keresés a vektortárban..")
            context = self.vector_store.search(question)
            print(f"Talált infók: {context}")

            response = self.generate_simulated_response(question,context)
        else:
            print(f"Közvetlen válasz:")
            response = self.generate_direct_response(question)

        return response

    def generate_simulated_response(self, question, context):
        return f"""
         Kérdés: {question}

        Talált információk: 
        {chr(10).join([f'- {doc[:80]}...' for doc in context])}

        Szimulált válasz: Ez egy szimulált válasz a talált információk alapján.
        """
    
    def generate_direct_response(self, question):
        
        return f"💬 Közvetlen válasz: '{question}'"