import getpass
import os, logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

from aiagent.config.logging_config import initialize_logger
from aiagent.routers.gemini_router import gemini_router

initialize_logger()

logger = logging.getLogger(__name__)
logger.info("aiagent service initializing")

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


def test_model():
  model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
  out = model.invoke("translate \"Good Morning!\" to Tamil")
  logger.info(f"model output {out.content}")

  messages = [
      SystemMessage("You are a translation engine. Only respond with the Tamil translation of the input. Do not add any explanation or commentary."),
      HumanMessage("Good Morning!"),
  ]

  for token in model.stream(messages):
      print(token.content, end="|")
    
    
async def on_startup():
    logger.info("running startup tasks")
    test_model()
    logger.info("model invoked successfully")
    
async def on_shutdown():
    logger.info("shutting down\n\n")
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    await on_startup()
    try:
        yield
    finally:
        await on_shutdown()
    

app = FastAPI(lifespan=lifespan, title="Ai Agent Service")

app.include_router(gemini_router)