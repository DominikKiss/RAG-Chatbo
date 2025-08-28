from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    question : str
    action : str
    response : str


def build_workflow(agent):

    def start_node(state):
        print("Start node")
        return state

    def decide_action_node(state):
        question = state["question"]
        action = agent.decide_action(question)
        state["action"] = action
        return state

    def search_knowledge_node(state):
        question = state["question"]
        context = agent.vector_store.search(question)
        response = agent.generate_simulated_response(question, context)
        state["response"] = response
        return state

    def direct_response_node(state):
        question = state["question"]
        response = agent.generate_direct_response(question)
        state["response"] = response
        return state

    # Gráf építése
    graph = StateGraph(state_schema=AgentState)
    graph.add_node("start", start_node)
    graph.add_node("decide_action", decide_action_node)
    graph.add_node("search_knowledge", search_knowledge_node)
    graph.add_node("direct_response", direct_response_node)

    graph.set_entry_point("start")
    graph.add_edge("start", "decide_action")

    graph.add_conditional_edges(
        "decide_action",
        lambda state: state["action"],
        {
            "search_knowledge": "search_knowledge",
            "direct_response": "direct_response"
        }
    )

    graph.add_edge("search_knowledge", END)
    graph.add_edge("direct_response", END)

    return graph.compile()
