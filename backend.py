import os
import json
import uuid
import asyncio
from threading import Lock
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.callbacks.manager import StdOutCallbackHandler, CallbackManager
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent, Tool

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore

from pinecone import Pinecone

from gsheet import GoogleSheetManager
from evaluation.llm_judge import LLMJudge
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langfuse.decorators import observe, langfuse_context
import requests

# ------------------------------------------------------------------------------
# Configuration and Initialization
# ------------------------------------------------------------------------------
load_dotenv()

# Load API keys and credentials
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

LANGFUSE_SECRET_KEY = os.getenv('LANGFUSE_SECRET_KEY')
LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY')
LANGFUSE_HOST = os.getenv('LANGFUSE_HOST')

# Initialize Google Sheets Manager for evaluations
eval_sheet_manager = GoogleSheetManager("MyMed_Agent_Evaluation", "testing (R3)")

# Initialize embeddings and LLM
embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model='text-embedding-3-small')
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Initialize Pinecone and set up vector store
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "mymed-allparams"
pinecone_index = pc.Index(INDEX_NAME)
print(f"Connected to Pinecone index '{INDEX_NAME}'.")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index, text_key="content")
vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Initialise Langfuse
langfuse = Langfuse(secret_key=LANGFUSE_SECRET_KEY, public_key=LANGFUSE_PUBLIC_KEY)

# # Initialize callback managers
callback_manager = CallbackManager([StdOutCallbackHandler()])
# langfuse_handler = CallbackHandler(
#     public_key=LANGFUSE_PUBLIC_KEY,
#     secret_key=LANGFUSE_SECRET_KEY,
#     host=LANGFUSE_HOST
# )

# ------------------------------------------------------------------------------
# Helper Classes and Functions
# ------------------------------------------------------------------------------
class QueryContextManager:
    def __init__(self):
        self._contexts = {}
        self._lock = Lock()

    def store_context(self, query_id: str, context: List[Dict]):
        with self._lock:
            self._contexts[query_id] = context

    def get_context(self, query_id: str) -> List[Dict]:
        return self._contexts.get(query_id, [])

    def clear_context(self, query_id: str):
        with self._lock:
            self._contexts.pop(query_id, None)

context_manager = QueryContextManager()
current_query_id = None  # Global to track the active query

def retrieve_from_vectorstore(query_text: str):
    global current_query_id
    # Embed the query text
    query_embedding = embedding_model.embed_query(query_text)

    # Query Pinecone for similar results
    pinecone_results = pinecone_index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    print(pinecone_results['matches'])

    # Store context for later evaluation
    context_manager.store_context(current_query_id, pinecone_results['matches'])

    # Combine retrieved text from metadata
    retrieved_texts = [result['metadata']['content'] for result in pinecone_results['matches']]
    return " ".join(retrieved_texts) if retrieved_texts else ""

# ------------------------------------------------------------------------------
# Agent and Tool Setup
# ------------------------------------------------------------------------------
# Define tools for the ReAct agent
retrieval_tool = Tool(
    name="pinecone_retriever",
    func=retrieve_from_vectorstore,
    description="Retrieves relevant documents from the Pinecone vector store"
)

tools = [retrieval_tool]  # Add health_status_tool to the list if required

# # Define the prompt template for the agent
# template = '''
# **Listen carefully, You are primarily programmed to communicate in English even if the retrieved documents are in Swedish.**
# **However, if user asks in another language, you must strictly respond in the same language as the users language.**

# Answer the following questions strictly using the given tools.

# {tools}

# Chat history:
# {chat_history}

# Use the following format:

# Question: the input question you must answer
# Thought: reason about the question and decide if it is relevant
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# Thought:
# - Use the **retrieval_tool** tool to retrieve information and context within your knowledge base.
# - If any tool returns a response starting with 'Final Answer:', **stop immediately** and use that as the response.
#     - Do not proceed with further tool usage or reasoning.

# Final Answer: the final answer to the original input question crafted like a storyline with steps if necessary. Include sources and links from the context ONLY.

# Begin!

# Question: {input}
# Thought: {agent_scratchpad}
# '''
langfuse_prompt = langfuse.get_prompt("ReAct RAG", label="english")
langfuse_prompt = langfuse.get_prompt("ReAct RAG", label="staging")
template = langfuse_prompt.prompt

prompt = PromptTemplate.from_template(template)

# Create the ReAct agent and memory
agent = create_react_agent(llm, tools, prompt)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Set up the Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callback_manager=callback_manager,
    handle_parsing_errors=True,
    memory=memory
)

# ------------------------------------------------------------------------------
# FastAPI Endpoints
# ------------------------------------------------------------------------------
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query/")
async def query_agent(query: QueryRequest):
    global current_query_id
    current_query_id = str(uuid.uuid4())

    # Create a new trace with your custom ID
    trace_client = langfuse.trace(
        id=current_query_id,
        metadata={"query": query.question}
    )

    # Get a Langchain callback handler from that trace
    my_handler = trace_client.get_langchain_handler(update_parent=True)

    print("CURRENT QUERY ID BACKEND", current_query_id)
    try:
        final_query = query.question  # If translation is needed, add that here

        # Invoke the ReAct agent
        response = agent_executor.invoke(
            {"input": final_query},
            config={"callbacks": [my_handler]},
        )

        print(f"Agent response: {response}")

        if not response or "Agent stopped" in response.get("output", ""):
            return {
                "response": "Sorry, I couldn't retrieve the information. Please try rephrasing your query as I can only provide health-related information."
            }

        # Retrieve context for evaluation
        contexts = context_manager.get_context(current_query_id)
        processed_contexts = []
        for ctx in contexts:
            if isinstance(ctx, dict):
                processed_contexts.append({
                    "source_url": ctx["metadata"].get("source_url", "N/A"),
                    "title": ctx["metadata"].get("title", "N/A"),
                    "content": ctx["metadata"].get("content", ""),
                    "score": ctx.get("score", "N/A")
                })
        contexts_str = json.dumps(processed_contexts, ensure_ascii=False, indent=2)

        # Evaluate the response using LLMJudge asynchronously
        llm_judge = LLMJudge()
        context_eval, groundedness_eval, answer_eval = await asyncio.gather(
            llm_judge.evaluate_context_relevance(query.question, contexts_str),
            llm_judge.evaluate_groundedness(query.question, contexts_str, response.get("output", "")),
            llm_judge.evaluate_answer_relevance(query.question, response.get("output", ""))
        )
        context_relevance_score, context_relevance_eval = context_eval
        groundedness_score, groundedness_eval = groundedness_eval
        answer_relevance_score, answer_relevance_eval = answer_eval

        # Log evaluations and context to Google Sheets
        eval_scores = {
            "llm_context_query_score": context_relevance_score,
            "llm_context_query_score_reasoning": context_relevance_eval,
            "llm_context_answer_score": groundedness_score,
            "llm_context_answer_score_reasoning": groundedness_eval,
            "llm_answer_query_score": answer_relevance_score,
            "llm_answer_query_score_reasoning": answer_relevance_eval,
        }
        eval_sheet_manager.log_agent_response(
            query=final_query,
            response=response.get("output", ""),
            contexts=contexts_str,
            eval_scores=eval_scores
        )

        # Return the final response
        return {"response": response.get("output", ""), 
                "trace_id": current_query_id  # Return the trace ID so the front end can store it
            }
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

class FeedbackRequest(BaseModel):
    trace_id: str
    query: str
    response: str
    feedback_value: str
    comment: str

@app.post("/feedback/")
def submit_feedback(feedback: FeedbackRequest):
    """
    Example: Accept user feedback, then call Langfuse /api/public/scores
    with dataType = "CATEGORICAL".
    """
    try:
        # Build the JSON payload for /api/public/scores.
        payload = {
            "traceId": feedback.trace_id,
            "name": "user_feedback",
            "value": feedback.feedback_value, # e.g., "correct", "incorrect", etc.
            "comment": feedback.comment,
            "dataType": "CATEGORICAL",        # important: we use a string value for categories
            "metadata": {
                "query": feedback.query,
                "response": feedback.response
            }
        }

        langfuse_scores_url = f"{LANGFUSE_HOST}/api/public/scores"
        print(f"Posting payload to {langfuse_scores_url}: {payload}")

        # Make the POST request to Langfuse
        resp = requests.post(langfuse_scores_url, json=payload, timeout=10, auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY))
        if resp.status_code == 200:
            print("Langfuse /scores accepted the feedback!")
        else:
            print(f"Langfuse /scores error: {resp.status_code}, {resp.text}")

        return {"message": "Feedback received successfully!"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error storing feedback: {str(e)}"
        )

# ------------------------------------------------------------------------------
# Server Startup
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=9009, reload=True)