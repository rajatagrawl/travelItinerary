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

class TravelRequest(BaseModel):
    prompt: str

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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

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
async def run_travel_agent(request: TravelRequest):
    try:
        # 1. Validate input
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # 2. Run your LangChain Agent Executor using the user's prompt
        result = agent_executor.invoke({"input": request.prompt})
        
        # 3. Extract the final text answer from the agent's output dictionary
        ai_itinerary = result.get("output", "Could not generate an itinerary.")

        # 4. Return the real AI response to your index.html frontend
        return {"response": ai_itinerary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# IMPORTANT: Keep /api/agent to align with your fetch request and Vercel rewrite
