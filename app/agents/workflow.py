from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage
from langgraph.graph import Graph, StateGraph
from langchain_openai import ChatOpenAI

# State definition
class GraphState(TypedDict):
    content: str
    entities: List[Dict]
    relationships: List[Dict]
    temp_graph: Dict
    changes: List[Dict]
    user_feedback: Optional[Dict]
    status: str

# Base Agent class
class BaseAgent:
    def __init__(self, llm):
        self.llm = llm

# Entity Resolution Agent
class LocalEntityResolutionAgent(BaseAgent):
    async def process(self, state: GraphState) -> GraphState:
        if not state.get('entities'):
            return state
            
        prompt = f"""
        Resolve and deduplicate these entities:
        {state['entities']}
        Consider similar names and aliases.
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        # Process response to resolve entities
        state['entities'] = resolved_entities
        state['status'] = 'entities_resolved'
        return state
    
# Entity Resolution Agent
class GlobalEntityResolutionAgent(BaseAgent):
    async def process(self, state: GraphState) -> GraphState:
        if not state.get('entities'):
            return state
            
        prompt = f"""
        Resolve and deduplicate these entities:
        {state['entities']}
        Consider similar names and aliases.
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        # Process response to resolve entities
        state['entities'] = resolved_entities
        state['status'] = 'entities_resolved'
        return state

# Graph Integration Agent
class GraphIntegrationAgent(BaseAgent):
    async def process(self, state: GraphState) -> GraphState:
        if not state.get('entities') or not state.get('relationships'):
            return state
            
        # Create temporary subgraph structure
        temp_graph = {
            'nodes': state['entities'],
            'relationships': state['relationships']
        }
        state['temp_graph'] = temp_graph
        state['status'] = 'graph_created'
        return state

# User Feedback Handler
class UserFeedbackHandler:
    def process_feedback(self, state: GraphState) -> GraphState:
        feedback = state.get('user_feedback')
        if not feedback:
            return state
            
        # Update graph based on user feedback
        state['temp_graph'] = updated_graph
        state['status'] = 'feedback_processed'
        return state

# Main Workflow
def create_workflow():
    # Initialize LLM
    llm = ChatOpenAI(temperature=0)
    
    # Initialize agents
    extractor = EntityExtractionAgent(llm)
    resolver = EntityResolutionAgent(llm)
    integrator = GraphIntegrationAgent(llm)
    feedback_handler = UserFeedbackHandler()

    # Create workflow graph
    workflow = StateGraph(GraphState)

    # Define edges
    workflow.add_node("extract", lambda x: extractor.process(x))
    workflow.add_node("resolve", lambda x: resolver.process(x))
    workflow.add_node("integrate", lambda x: integrator.process(x))
    workflow.add_node("handle_feedback", lambda x: feedback_handler.process_feedback(x))

    # Define conditional edges
    def should_get_feedback(state: GraphState) -> str:
        if state['status'] == 'graph_created':
            return "handle_feedback"
        return "end"

    # Connect nodes
    workflow.add_edge("extract", "resolve")
    workflow.add_edge("resolve", "integrate")
    workflow.add_edge("integrate", should_get_feedback)
    workflow.add_edge("handle_feedback", "resolve")
    
    workflow.set_entry_point("extract")
    
    return workflow.compile()

# Usage
async def process_document(content: str):
    workflow = create_workflow()
    
    initial_state = {
        "content": content,
        "entities": [],
        "relationships": [],
        "temp_graph": {},
        "changes": [],
        "user_feedback": None,
        "status": "started"
    }
    
    async for state in workflow.astream(initial_state):
        # Emit state updates for frontend
        if state['status'] == 'entities_extracted':
            await notify_frontend("Entities extracted", state['entities'])
        elif state['status'] == 'entities_resolved':
            await notify_frontend("Entities resolved", state['changes'])
        elif state['status'] == 'graph_created':
            await notify_frontend("Graph ready for review", state['temp_graph'])
            # Wait for user feedback
            user_feedback = await get_user_feedback()
            state['user_feedback'] = user_feedback