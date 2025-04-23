from django.contrib.auth.models import User
from rest_framework.decorators import api_view,permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
import asyncio
import json
import traceback
from rest_framework import status
from rest_framework_simplejwt.views import TokenObtainPairView
from .models import notes
from django.http import JsonResponse
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
# from .services.langchain_service import generate_content_pipeline
from .services.langchain_service import ContentRequest, StoryIterationChain
from .serializers import notesSerializers
import requests
from asgiref.sync import async_to_sync
from django.http import JsonResponse
import base64
import io
import cohere
import logging
from dotenv import load_dotenv 
import os
from pydantic import ValidationError

# ------------------------ LOGGING ------------------------ #
logger = logging.getLogger(__name__)
load_dotenv()

# ------------------------ APIs ------------------------ #
co = cohere.Client(os.getenv("CO_API_KEY"))
    
langchain_service = None
story_chain_service = None

async def get_story_chain_service():
    """Async factory function for StoryIterationChain"""
    try:
        # Use environment variables for each service
        image_service_url = os.getenv("IMAGE_SERVICE_URL", "http://image-service:5000")
        voice_service_url = os.getenv("VOICE_SERVICE_URL", "http://voice-service:5000")
        whisper_service_url = os.getenv("WHISPER_SERVICE_URL", "http://whisper-service:5000")
        
        service = StoryIterationChain(
            colab_url=f"{image_service_url}/generate-image", 
            voice_url=f"{voice_service_url}/generate_sound", 
            whisper_url=f"{whisper_service_url}/process_audio"
        )
        logger.info("StoryIterationChain service created successfully")
        return service
    except Exception as e:
        logger.error(f"Error creating StoryIterationChain service: {str(e)}")
        raise

@csrf_exempt
async def generate_content(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Invalid request method."}, status=405)
        
    try:
        data = json.loads(request.body)
            
        content_request = ContentRequest(
            prompt=data.get("prompt"),
            genre=data.get("genre", "cyberpunk"),
            iterations=data.get("iterations", 4),
            backgroundVideo=data.get("backgroundType", "urban"),
            backgroundMusic=data.get("musicType", "synthwave"),
            voiceType=data.get("voiceType", "male"),
            subtitleColor=data.get("subtitleColor", "#ff00ff")
        )
        
        logger.info(f"Content request: {content_request}")
        
        service = await get_story_chain_service()
        result = await service.generate_content_pipeline(content_request)
        
        response_data = {
            "success": True,
            "video_data": result["video_data"],
            "content_type": result["content_type"],
            "metrics": result["metrics"]
        }
        
        logger.info("Returning video response")
        return JsonResponse(response_data, status=200)
            
    except Exception as e:
        error_msg = f"Content generation error: {str(e)}"
        logger.error(error_msg)
        return JsonResponse({"error": error_msg}, status=500)




# ---------------------- AUTH AND USER MANAGEMENT ----------------------
class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        # Add custom claims
        token['username'] = user.username
        token['password'] = user.password
        # ...
        return token

class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer

@api_view(['GET'])
def getRoutes(request):
    routes = [
        '/api/token',
        '/api/token/refresh',
    ]
    return Response(routes)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def getUserDetails(request):
    user = request.user
    user_data = {
        'id': user.id,
        'username': user.username,
    }
    return Response(user_data)

@api_view(['POST'])
def create(request):
    data = request.data
    username = data.get("username", "").lower()
    password = data.get("password", "")
    if User.objects.filter(username=username).exists():
        return Response({"error": "USER ALREADY EXISTS"}, status=status.HTTP_400_BAD_REQUEST)
    try:
        user = User.objects.create_user(username=username,password=password)
        user.save()
        return Response({"message": "User created successfully"}, status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def getNotes(request):
    user = request.user
    Notes = user.notes_set.all()  
    serializer = notesSerializers(Notes, many=True)
    return Response(serializer.data)
