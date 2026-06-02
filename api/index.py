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
    """Fetch top local attractions and estimated visit times for any given city dynamically."""
    try:
        # Spin up a lightweight, deterministic backend instance to fetch real-world data points
        data_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        response = data_llm.invoke(
            f"List 4 to 6 top popular local attractions and landmarks for {city} along with a recommended visit duration for each in parentheses. "
            f"Keep the output brief, factual, and strictly on one line. Example: Eiffel Tower (2 hours), Louvre Museum (4 hours)"
        )
        return response.content
    except Exception as e:
        # Safe fallback if the API network call experiences a temporary hiccup
        return f"Popular local landmarks, historic quarters, and cultural markets in {city}."

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
    ("system", (
        "You are an expert Travel Agent. You MUST use the provided tools to fetch attractions and budgets.\n\n"
        "CRITICAL FORMATTING RULES:\n"
        "1. You must organize the attractions found into a clear, day-by-day itinerary (e.g., Day 1:, Day 2:).\n"
        "2. Display the budget options clearly using the tool output.\n"
        "3. Output ONLY the itinerary and the budget. Do NOT include any concluding remarks, questions, conversational filler, or offers for further help at the end of your response."
    )),
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
