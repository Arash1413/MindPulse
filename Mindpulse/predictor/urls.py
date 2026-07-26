from django.contrib.admin import views
from django.urls import path
from .views import chatbot_api, export_pdf_report, predict_wellbeing
from .views import predict_wellbeing, compare_models
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', predict_wellbeing, name='predict'),
    path('compare/', compare_models, name='compare'),
    path('export-pdf/', export_pdf_report, name='export_pdf'),
    path('api/chatbot/', chatbot_api, name='chatbot_api'), # 👈 مسیر چت‌بات
    path('history/', views.history_view, name='history'),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('register/', views.register_view, name='register'), # 👈 اضافه شد
]