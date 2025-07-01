import logging
from aiagent.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

class GeminiController:
    
    def __init__(self, gemini_service:GeminiService ):
        self.service = gemini_service
        
    def translate_to(self, message:str, lang:str) -> str:
        return self.service.translateTo(message, lang)
    
    def invoke(self, message:str) -> str :
        return self.service.invoke(message)