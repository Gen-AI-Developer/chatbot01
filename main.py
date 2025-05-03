import json
from textwrap import indent
from time import sleep
import chainlit as cl
import dotenv
import os
from openai.types.responses import ResponseTextDeltaEvent

from agents import Agent, Runner, set_tracing_disabled
from agents import enable_verbose_stdout_logging
from agents.extensions.models.litellm_model import LitellmModel

#setting verbose logging
# This will enable verbose logging to stdout
# This is useful for debugging and understanding the flow of the program
enable_verbose_stdout_logging()
#setting tracing disabled
set_tracing_disabled(True)
# Load environment variables from .env file
dotenv.load_dotenv()
# Set the GEMINI_API_KEY key from environment variable and getting model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL: str = os.getenv("MODEL") or ""

web_developer_agent: Agent = Agent(
      name="Web Developer Agent",
      instructions="You are expert on Full Stack Web Developer your Tech stack in Nextjs, FastAPI, MongoDB, Nodejs, TailwindCSS.",
      model=LitellmModel(model=MODEL,api_key=GEMINI_API_KEY),
      handoff_description="Web Developer Expert",
    )
mobile_developer_agent: Agent = Agent(
      name="Mobile Developer Agent",
      model=LitellmModel(model=MODEL,api_key=GEMINI_API_KEY),
      instructions="You are expert on Mobile Application Developer Expert. Your techstack in Kotlin/java and Firebase",
      handoff_description="Mobile Application Developer Expert in Java Kotlin/ XML",
    )
dev_ops_agent: Agent = Agent(
      name="Dev OPS Agent",
      model=LitellmModel(model=MODEL,api_key=GEMINI_API_KEY),
      instructions="You are expert on DevOPS and CI/CD",
    )
openai_agent: Agent = Agent(
      name="OpenAI Agent",
      model=LitellmModel(model=MODEL,api_key=GEMINI_API_KEY),
      instructions="You are expert on Open AI Agent SDK",
    )
agentic_ai_dev_agent: Agent = Agent(
      name="Agentic AI Developer Agent",
      model=LitellmModel(model=MODEL,api_key=GEMINI_API_KEY),
      instructions="You are expert on Agentic AI Development, System Design.",
      handoff_description="You are expert on Agentic AI Development",
      tools=[dev_ops_agent.as_tool(
          tool_name="dev_OPS_Agent",
          tool_description="You are expert on DevOPS and CI/CD",
      ),openai_agent.as_tool(
          tool_name="openAI_Agent",
          tool_description="You are expert on Open AI Agent SDK",
      )]
    )

triage_agent : Agent = Agent(
      name="Triage Agent",
      instructions="You Are a Triage Panacloud Agent, you pass question/query to relavent Agent",
      model=LitellmModel(model=MODEL,api_key=GEMINI_API_KEY),
      handoffs=[web_developer_agent,mobile_developer_agent,agentic_ai_dev_agent],
    )
@cl.on_chat_start
async def on_chat_start():
    
    # Send a message to the user
    cl.user_session.set("chat_history", [])
    await cl.Message("Hi, How I can Help you.").send()


    # history = cl.user_session.get('query') or []

@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="Thinking...")
    await msg.send()

    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})
    response = Runner.run_streamed(triage_agent, history)

    full_response = ""

    async for event in response.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            full_response += event.data.delta
            msg.content = full_response
            sleep(1)
            await msg.update()

    history.append({"role": "assistant", "content": response.final_output})


@cl.on_chat_end
async def on_chat_end():
    history = cl.user_session.get("chat_history") or []
    with open("chat_history.json","w") as f:
        json.dump(history,f,indent=2)
    print("Chat History Saved")