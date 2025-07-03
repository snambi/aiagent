import logging

from fastapi import APIRouter, Depends

from aiagent.controllers.gemini_controller import GeminiController
from aiagent.services.gemini_service import GeminiService
from aiagent.models.gemini_input import TranslationRequest

logger = logging.getLogger(__name__)

gemini_router = APIRouter(prefix="/gemini", 
                          tags=["translate"], 
                          responses={404:{"description": "Not Found"}}
                          )

gemini_controller:GeminiController = None


def get_gemini_service():
    return GeminiService()


def get_gemini_controller(service:GeminiService = Depends(get_gemini_service) ):
    global gemini_controller
    if( gemini_controller is None ):
        gemini_controller = GeminiController(service)
        
    return gemini_controller


@gemini_router.post("/translate", response_model=str )
async def translate_to( request:TranslationRequest, controller:GeminiController = Depends(get_gemini_controller) ):
    return controller.translate_to(request.text, request.language)

