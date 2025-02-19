# uvicorn backend:app --reload --host 0.0.0.0 --port 9009

# consider
## uvicorn agent_run:app --log-config logging.conf

# from googletrans import Translator
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from threading import Thread, Lock
import json
import uuid
import asyncio

import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.callbacks.manager import StdOutCallbackHandler, CallbackManager
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent, Tool

# from langchain_google_genai import ChatGoogleGenerativeAI

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex

from dotenv import load_dotenv

from pinecone import Pinecone

from gsheet import GoogleSheetManager
from evaluation.llm_judge import LLMJudge

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# translator = Translator()

# Load API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

LANGFUSE_SECRET_KEY = os.getenv('LANGFUSE_SECRET_KEY')
LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY')
LANGFUSE_HOST = os.getenv('LANGFUSE_HOST')

# eval_sheet_manager = GoogleSheetManager("MyMed_Agent_Evaluation", "translated-4o-mini (R3)")
eval_sheet_manager = GoogleSheetManager("MyMed_Agent_Evaluation", "translated-4o-mini (R3) (sv)")

embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model='text-embedding-3-small')

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)  # Assuming gpt-4o-mini model in use
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)

# Set up Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "mymed-allparams"

# Connect to the Pinecone index
pinecone_index = pc.Index(index_name)
print(f"Connected to Pinecone index '{index_name}'.")

vector_store = PineconeVectorStore(pinecone_index=pinecone_index, text_key="content")

vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Callback for printing thought process
callback_manager = CallbackManager([StdOutCallbackHandler()])

# Initialize Langfuse
langfuse = Langfuse(secret_key=LANGFUSE_SECRET_KEY, public_key=LANGFUSE_PUBLIC_KEY)

langfuse_handler = CallbackHandler(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)

current_query_id = None

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

# 1. RETRIEVAL tool that fetches similar documents from Pinecone
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
    context_manager.store_context(current_query_id, pinecone_results['matches'])

    # Retrieve and combine text
    retrieved_texts = [result['metadata']['content'] for result in pinecone_results['matches']]

    if retrieved_texts:
        return " ".join(retrieved_texts)

with open('/Users/briannoel/Desktop/mymed-combined/data/ranges.json') as f:
    limits_json = json.load(f)

# 2. health data and its categories
def evaluate_health_status(health_values, limits_data):
    evaluated_health = []

    for param in limits_data:
        param_name = param["name"]
        user_value = health_values.get(param_name, None)

        if user_value is not None:
            for range_def in param["ranges"]:
                if range_def["min"] <= user_value <= range_def["max"]:
                    evaluated_health.append({
                        "parameter": param_name,
                        "value": user_value,
                        "status": range_def["description"]
                    })
                    break  # Stop checking further ranges once matched

    return evaluated_health

# Function to evaluate user health data
def retrieve_health_status(user_health_data):
    evaluated_health_status = evaluate_health_status(user_health_data, limits_json)
    return json.dumps(evaluated_health_status, indent=2)

retrieval_tool = Tool(
    name="pinecone_retriever",
    func=lambda query_text: retrieve_from_vectorstore(query_text),
    description="Retrieves relevant documents from the Pinecone vector store"
)

# Create a Tool for retrieving health status
health_status_tool = Tool(
    name="health_status_evaluator",
    func=lambda user_data: retrieve_health_status(user_data),
    description="Evaluates user health parameters against predefined limits and categorizes them as Low, Normal, Moderate, or High."
)

tools = [retrieval_tool]

# Create the Prompt

template = '''
    **Listen carefully, You are primarily programmed to communicate in Swedish since the retrieved documents are in Swedish.**
    **However, if user asks in another language, you must strictly respond in the same language as the users language.**

    Answer the following questions strictly using the given tools.

    {tools}

    Chat history:
    {chat_history}

    Use the following format:

    Question: the input question you must answer
    Thought: reason about the question and decide if it is relevant
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    Thought:
    - Use the **retrieval_tool** tool to retrieve information and context within your knowledge base.
    - If any tool returns a response starting with 'Final Answer:', **stop immediately** and use that as the response.
        - Do not proceed with further tool usage or reasoning.

    Final Answer: the final answer to the original input question crafted like a storyline with steps if necessary. Include sources and links from the context ONLY.

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
'''

# user_health_data = {
#     "Datum": "2/1/2025 5:00",
#     "Weight": 70,
#     "BMI": 20,
#     "Pulse": 60,
#     "Systolic": 110,
#     "Diastolic": 70,
#     "BloodSugarShortTerm": 3,
#     "BloodSugarLongTerm": 35,
#     "Steps": 11000,
#     "Sleep": 26000,
#     "ASAT": 0.5,
#     "ALAT": 0.5,
#     "EnergyLevel": 6.5,
#     "Mood": 6.5,
#     "Stress": 1,
#     "Anxiety": 1,
#     "Depression": 1,
#     "Cholesterol": 3,
#     "RespiratoryRate": 13,
#     "LDLCholesterol": 3,
#     "HDLCholesterol": 1,
#     "THSThyroidStimulatingHormone": 2,
#     "T3Triiodothyronine": 5,
#     "T4Thyroxine": 15,
#     "Estrogen": 80,
#     "Testosterone": 10,
#     "NaSodium": 140,
#     "KPotassium": 4,
#     "Creatinine": 50,
#     "PSAProstateSpecificAntigen": 1,
#     "WaterThrowingBesvVAS": 0.5
# }

# template = '''

#     Answer the following questions strictly using the given tools.

#     {tools}

#     Chat history:
#     {chat_history}

#     Use the following format:

#     Question: the input question you must answer
#     Thought: reason about the question and decide if it is relevant
#     Action: the action to take, should be one of [{tool_names}]
#     Action Input: the input to the action
#     Observation: the result of the action
#     Thought:
#     - Use the **retrieval_tool** tool to retrieve information and context within your knowledge base.
#     - If any tool returns a response starting with 'Final Answer:', **stop immediately** and use that as the response.
#         - Do not proceed with further tool usage or reasoning.

#     Final Answer: the final answer to the original input question crafted like a storyline with steps if necessary.

#     Begin!

#     Question: {input}
#     Thought: {agent_scratchpad}
# '''

prompt = PromptTemplate.from_template(template)

# Step 3: Initialize the ReAct Agent
agent = create_react_agent(llm, tools, prompt)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Step 4: Set up the Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, callback_manager=callback_manager, handle_parsing_errors=True, memory=memory)

# Step 5: Run a Query through the ReAct Agent
# Request model for FastAPI
class QueryRequest(BaseModel):
    question: str

USER_PARAM = """
This is my health data that you should take into account when answering my questions:

Blodtryck {Systolic}/{Diastolic} mmHg
Sömn {Sleep} timmar
BMI {BMI}
Puls {Pulse} bpm
B-glukos {BloodSugarShortTerm} mmol/L
HbA1c {BloodSugarLongTerm} mmol/mol
Steg {Steps}/d
Kolesterol {Cholesterol} mmol/L
LDL {LDLCholesterol} mmol/L
HDL {HDLCholesterol} mmol/L
TSH {THSThyroidStimulatingHormone} mIE/L
T3 {T3Triiodothyronine} pmol/L
T4 {T4Thyroxine} pmol/L
ASAT {ASAT} µkat/L
ALAT {ALAT} µkat/L
Kreatinin {Creatinine} µmol/L
PSA {PSAProstateSpecificAntigen} µg/L
Na {NaSodium} mmol/L
K {KPotassium} mmol/L
Testosteron {Testosterone} nmol/L
Östrogen {Estrogen} pmol/L
Andningsfrekvens {RespiratoryRate} andetag/min
Energilevel {EnergyLevel}/10
Humör {Mood}/10
Stress {Stress} nivå
Ångest {Anxiety} nivå
Depression {Depression} nivå
Vattenkastning {WaterThrowingBesvVAS} VAS-skala
""".format(
    Systolic=140,
    Diastolic=70,
    Sleep=26000 / 3600,  # Convert from seconds to hours
    BMI=20,
    Pulse=60,
    BloodSugarShortTerm=3,
    BloodSugarLongTerm=35,
    Steps=11000,
    Cholesterol=3,
    LDLCholesterol=8,
    HDLCholesterol=8,
    THSThyroidStimulatingHormone=2,
    T3Triiodothyronine=5,
    T4Thyroxine=15,
    ASAT=0.5,
    ALAT=0.5,
    Creatinine=50,
    PSAProstateSpecificAntigen=1,
    NaSodium=140,
    KPotassium=4,
    Testosterone=10,
    Estrogen=80,
    RespiratoryRate=13,
    EnergyLevel=6.5,
    Mood=6.5,
    Stress=1,
    Anxiety=1,
    Depression=1,
    WaterThrowingBesvVAS=0.5
)

# API Endpoint for querying the ReAct Agent
@app.post("/query/")
async def query_agent(query: QueryRequest):
    global current_query_id
    current_query_id = str(uuid.uuid4())
    try:
        # Pass the input query to the agent executor
        # translated_result = await translator.translate(query.question, src="en", dest="sv")
        # translated_query = translated_result.text

        # final_query = translated_query if translated_query != query.question else query.question

        final_query = query.question

        response = agent_executor.invoke({"input":  final_query}, config={"callbacks": [langfuse_handler]})

        # response = agent_executor.invoke({"input":  USER_PARAM + "\n" + final_query}, config={"callbacks": [langfuse_handler]})

        # translated_result_response = await translator.translate(response.get("output"), src="sv", dest="en")
        # translated_response = translated_result_response.text

        print(f"Agent response: {response}")
        # print(f"Agent translated response: {translated_response}")

        # Ensure there's a valid output
        if not response or "Agent stopped" in response.get("output", ""):
            return {
                "response": "Sorry, I couldn't retrieve the information. Please try rephrasing your query as I can only provide health-related information from 1177."
            }
        
        # Run Evaluations
        llm_judge = LLMJudge()

        # Get contexts for evaluation
        contexts = context_manager.get_context(current_query_id)

        # print("Context before processing: " , contexts)

        # Extract only relevant metadata from contexts
        processed_contexts = []
        for ctx in contexts:
            if hasattr(ctx, 'metadata') and hasattr(ctx, 'score'):
                processed_contexts.append({
                    "source_url": ctx["metadata"].get("source_url", "N/A"),
                    "title": ctx["metadata"].get("title", "N/A"),
                    "content": ctx["metadata"].get("content", ""),
                    "score": ctx.get("score", "N/A")  # Include similarity score
                })
        # print("Processed context: " , processed_contexts)

        # Convert processed contexts to JSON string
        contexts_str = json.dumps(processed_contexts, ensure_ascii=False, indent=2)

        # print("Final contexts string: " , contexts_str)

        # context_eval, groundedness_eval, answer_eval = await asyncio.gather(
        #     llm_judge.evaluate_context_relevance(final_query, contexts_str),
        #     llm_judge.evaluate_groundedness(final_query, contexts_str, translated_response),
        #     llm_judge.evaluate_answer_relevance(final_query, translated_response)
        #     )
        
        # print(context_eval)

        context_eval, groundedness_eval, answer_eval = await asyncio.gather(
            llm_judge.evaluate_context_relevance(query.question, contexts_str),
            llm_judge.evaluate_groundedness(query.question, contexts_str, response.get("output", "")),
            llm_judge.evaluate_answer_relevance(query.question, response.get("output", ""))
        )

        context_relevance_score, context_relevance_eval = context_eval
        groundedness_score, groundedness_eval = groundedness_eval
        answer_relevance_score, answer_relevance_eval = answer_eval

        # context_relevance_score, context_relevance_eval = llm_judge.evaluate_context_relevance(query.question, contexts_str)
        # groundedness_score, groundedness_eval = llm_judge.evaluate_groundedness(query.question, contexts_str, response.get("output", ""))
        # answer_relevance_score, answer_relevance_eval = llm_judge.evaluate_answer_relevance(query.question, response.get("output", ""))

        # Format evaluation scores
        eval_scores = {
            "llm_context_query_score": context_relevance_score,
            "llm_context_query_score_reasoning": context_relevance_eval,

            "llm_context_answer_score": groundedness_score,
            "llm_context_answer_score_reasoning": groundedness_eval,

            "llm_answer_query_score": answer_relevance_score,
            "llm_answer_query_score_reasoning": answer_relevance_eval,
        }
    
        # Log to sheets
        eval_sheet_manager.log_agent_response(
            query=final_query,
            response=response.get("output", ""),
            contexts=contexts_str,
            eval_scores=eval_scores
        )

        processed_contexts = []
        
        # Extract and return the final response
        return {"response": response.get("output", ""),}
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

# Data model for feedback submission
class FeedbackRequest(BaseModel):
    query: str
    response: str
    feedback_score: int  # 1-5 rating
    trace_id: str  # Unique Trace ID

@app.post("/feedback/")
async def submit_feedback(feedback: FeedbackRequest):
    """Attach user feedback to Langfuse trace and store it in Google Sheets."""
    try:
        # Attach feedback to the existing trace in Langfuse
        langfuse.add_trace_feedback(
            trace_id=feedback.trace_id,
            score=feedback.feedback_score,
            comment=f"User rated {feedback.feedback_score}/5"
        )

        return {"message": "Feedback received successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing feedback: {str(e)}")
    
# # Start FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9009)