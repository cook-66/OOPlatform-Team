from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name="index"),
    path('api/infer/', views.api_run_inference, name="api_infer"),
    path('api/explain/', views.api_generate_explanation, name="api_explain"),
    path('api/today-cost/', views.api_today_cost, name="api_today_cost"),
    path('api/history/', views.api_history, name="api_history"),
    path('api/trend/', views.api_trend, name="api_trend"),
    path('api/record/<int:record_id>/', views.api_record_detail, name="api_record_detail"),
    path('api/chat/', views.api_chat_assistant, name="api_chat"),
]
