import logging, os, getpass

from langchain.chat_models import init_chat_model

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
        
  
    
    def stream(self, input:str):
        