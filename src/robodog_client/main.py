#!/usr/bin/env python

import asyncio
import os
import json
from typing import Any
from dotenv import load_dotenv
from .utils.prompt import system_prompt

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  

load_dotenv()

class CustomEncoder(json.JSONEncoder):
  def default(self, o: Any):
    if hasattr(o, "content"):
      return {"type": o.__class__.__name__, "content": o.content}
    return super().default(o)

llm = ChatGoogleGenerativeAI(
  model="gemini-2.0-flash",
  temperature=0,
  max_retries=2,
  google_api_key=os.getenv("GOOGLE_API_KEY")
)

client = MultiServerMCPClient(
  {
    "robodog": {
      "transport": "streamable_http",
      "url": "http://127.0.0.1:8000/mcp"
    },
  }
)

async def run_agent():

  print("🔌 Connecting to MCP servers...")
  tools = await client.get_tools()
  print(f"🔧 Loaded {len(tools)} MCP tools.")

  agent = create_agent(
    llm, 
    tools, 
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(), 
  )

  print("🚀 LangChain MCP Client started. Type 'quit' to exit.")

  while True:
    query = input("\nQuery: ").strip()
    if query.lower() == "quit":
      print("👋 Exiting MCP Client.")
      break

    response = await agent.ainvoke(
      {"messages": query}, 
      {"configurable": {"thread_id": "1"}}, 
    )

    try:
      formatted = json.dumps(response, indent=2, cls=CustomEncoder)
    except:
      formatted = str(response)

    print("\nResponse:")
    print(formatted)

def main():
  asyncio.run(run_agent())
