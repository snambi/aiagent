import logging, os, getpass
from typing import AsyncGenerator

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

"""

"""
class GeminiService:
    
    model = None
    
    def __init__(self):
        logger.info("starting gemini service")
        
        if not os.environ.get("GOOGLE_API_KEY"):
          os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
          
        self.model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        v = self.model.invoke("Hello, world!")
        
        logger.info("output from gemini: "+ v.content)
        
  
    
    def invoke(self, input:str) -> str :
        output = self.model.invoke("Hello, world!")
        
        logger.debug("input: {input} , output: {output.content} ")
        
        return output.content
    
    def translateTo(self, input:str, language:str ) -> str:
        
        messages = [
                    SystemMessage(f"You are a translation engine. Only respond with the {language} translation of the input. Do not add any explanation or commentary."),
                    HumanMessage(input),
                ]

        output = self.model.invoke(messages)
        
        return output.content
    

    async def translateToInStream(self, input:str, language:str) -> AsyncGenerator[str, None]:
        
        logger.debug(f"translate {input} to {language}")
        
        messages = [
            SystemMessage(f"You are a translation engine. Only respond with the {language} translation of the input. Do not add any explanation or commentary."),
            HumanMessage(input),
        ]
        
        for chunk in self.model.stream(messages):
            if chunk.content :
                yield chunk.content.encode("utf-8")