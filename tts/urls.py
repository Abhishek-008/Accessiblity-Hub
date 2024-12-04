from django.urls import path
from tts import views

urlpatterns = [
    path("",views.index,name='home'),
    path('text-to-speech/', views.text_to_speech_view, name='text_to_speech'),
    path('texttospeech/', views.texttospeech, name='texttospeech'),
    path('speech-to-text/', views.speech_recognition_view, name='speech_to_text'),
    path('gesture/', views.gesture_recognition_page, name='gesture_recognition'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('start_detection/', views.start_gesture_detection, name='start_gesture_detection'),
    path('stop_detection/', views.stop_gesture_detection, name='stop_gesture_detection'),
    path('detect/', views.detect_objects, name='detect_objects'),

]
