from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name="index"),
    path('api/infer/', views.api_run_inference, name="api_infer"),
    path('api/explain/', views.api_generate_explanation, name="api_explain"),
]
