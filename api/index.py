import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# -------------------------------------------------------------
# 🔒 CORS Setup (Allows your HTML frontend to talk to your API)
# -------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows requests from any origin (perfect for Vercel deployment)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# 🛠️ Define the Tools (Functions the AI Agent can execute)
# -------------------------------------------------------------

@tool
def get_attractions(city: str) -> str:
    """Fetch top local attractions and estimated visit times for a given city."""
    database = {
        "paris": "Eiffel Tower (2 hours), Louvre Museum (4 hours), Montmartre (3 hours)",
        "tokyo": "Shibuya Crossing (1 hour), Senso-ji Temple (2 hours), Akihabara (3 hours)",
    }
    return database.get(city.lower(), "Local Markets (2 hours), Central Park (2 hours)")

@tool
def calculate_budget(days: int, style: str) -> str:
    """Calculate the total estimated cost in USD based on days and travel style ('budget' or 'luxury')."""
    per_day = 300 if style.lower() == "luxury" else 70
    total = days * per_day
    return f"The estimated total cost for a {style} trip lasting {days} days is ${total} USD."

tools = [get_attractions, calculate_budget]

# -------------------------------------------------------------
# 🤖 Initialize the Agent
# -------------------------------------------------------------

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert, proactive Travel Agent. You MUST use the provided tools to fetch attractions and calculate budgets. Do not make up prices or guess attractions without using tools."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"), 
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# -------------------------------------------------------------
# 🌐 Web Endpoints
# -------------------------------------------------------------

class UserQuery(BaseModel):
    prompt: str

@app.post("/api/agent")
def run_travel_agent(query: UserQuery):
    try:
        response = agent_executor.invoke({"input": query.prompt})
        return {"response": response["output"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
