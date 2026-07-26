from django.db import models
from django.contrib.auth.models import User

class WellbeingAssessment(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    
    # ویژگی‌های اصلی نمونه
    age = models.FloatField(default=20)
    sex = models.FloatField(default=1) # مثلا 1 برای مرد، 2 برای زن
    resilience = models.FloatField(default=50) # res
    
    # مکانیسم‌های مقابله‌ای اصلی (مهم‌ترین‌ها)
    active_coping = models.FloatField(default=3.0)
    planning = models.FloatField(default=3.0)
    positive_reframing = models.FloatField(default=3.0)
    acceptance = models.FloatField(default=3.0)
    self_blame = models.FloatField(default=2.0)
    
    # نمره پیش‌بینی شده نهایی توسط مدل Hybrid ML
    predicted_wellbeing = models.FloatField()

    def __str__(self):
        return f"Assessment {self.id} - Score: {self.predicted_wellbeing:.2f}"


class AssessmentHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='assessments')
    created_at = models.DateTimeField(auto_now_add=True)
    score = models.FloatField()
    
    # ذخیره داده‌های راداری به‌صورت JSON یا فیلدهای مجزا
    radar_data = models.JSONField(help_text="اطلاعات متغیرهای راداری")

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.score} - {self.created_at.strftime('%Y-%m-%d')}"