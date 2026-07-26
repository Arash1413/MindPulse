import os
import json
import joblib
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
import base64
from io import BytesIO
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from google import genai
from django.contrib.auth.decorators import login_required
from .models import AssessmentHistory
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'predictor', 'ml', 'wellbeing_hybrid_pipeline.pkl')

pipeline = joblib.load(MODEL_PATH)
imputer = pipeline['imputer']
scaler = pipeline['scaler']
lgbm_model = pipeline['lgbm']
elastic_model = pipeline['elastic']
feature_names = pipeline['feature_names']

# views.py


def register_view(request):
    if request.user.is_authenticated:
        return redirect('history')  
        
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  
            return redirect('history')
    else:
        form = UserCreationForm()
        
    return render(request, 'register.html', {'form': form})


def predict_wellbeing(request):
    if request.method == 'POST':
        age = float(request.POST.get('age', 22))
        sex = float(request.POST.get('sex', 1))
        res = float(request.POST.get('res', 50))
        
        active_coping = float(request.POST.get('active_coping', 3.0))
        planning = float(request.POST.get('planning', 3.0))
        positive_reframing = float(request.POST.get('positive_reframing', 3.0))
        acceptance = float(request.POST.get('acceptance', 3.0))
        emotional_support = float(request.POST.get('emotional_support', 3.0))
        humor = float(request.POST.get('humor', 2.5))
        self_blame = float(request.POST.get('self_blame', 2.0))
        denial = float(request.POST.get('denial', 1.5))
        behavioral_disengagement = float(request.POST.get('behavioral_disengagement', 1.5))
        self_distraction = float(request.POST.get('self_distraction', 2.0))

       
        input_dict = {col: 3.0 for col in feature_names}
        input_dict['age'] = age
        input_dict['sex'] = sex
        input_dict['res'] = res
        input_dict['Active_coping'] = active_coping
        input_dict['Planning'] = planning
        input_dict['Positive_reframing'] = positive_reframing
        input_dict['Acceptance'] = acceptance
        input_dict['Self_blame'] = self_blame
        input_dict['Emotional_support'] = emotional_support
        input_dict['Humor'] = humor
        input_dict['Denial'] = denial
        input_dict['Behavioral_disengagement'] = behavioral_disengagement
        input_dict['Self_distraction'] = self_distraction

        input_df = pd.DataFrame([input_dict])[feature_names]

       
        input_imp = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imp)

        pred_lgbm = lgbm_model.predict(input_imp)[0]
        pred_elastic = elastic_model.predict(input_scaled)[0]

        raw_score = (0.85 * pred_lgbm) + (0.15 * pred_elastic)
        
        
        mean_anchor = 39.0
        stretch_factor = 2.5  
        
        adjusted_score = mean_anchor + (raw_score - mean_anchor) * stretch_factor
        
        
        final_score = round(float(np.clip(adjusted_score, 10, 90)), 2)

        
        recommendations = []
        if self_blame >= 3.0:
            recommendations.append("خودسرزنش‌گری بالا: تمرین‌های شفقت ورزیدن به خود (Self-Compassion) را انجام دهید.")
        if denial >= 2.5 or behavioral_disengagement >= 2.5:
            recommendations.append("اجتناب و تسلیم‌شدگی: مواجهه آگاهانه با چالش‌ها به جای فرار از مشکل.")
        if res < 50:
            recommendations.append("تاب‌آوری پایین: تقویت شبکه حمایت اجتماعی و تنظیم اهداف کوچک قابل دستیابی.")
        if active_coping < 2.5 and planning < 2.5:
            recommendations.append("حل مسئله: استفاده بیشتر از تکنیک‌های برنامه‌ریزی و اقدام فعالانه.")

        
        radar_data = {
            'مقابله فعالانه': active_coping,
            'برنامه‌ریزی': planning,
            'نگاه مثبت': positive_reframing,
            'پذیرش': acceptance,
            'حمایت عاطفی': emotional_support,
            'شوخ‌طبعی': humor,
            'خودسرزنش‌گری': self_blame,
            'انکار و اجتناب': denial,
        }

        
        if request.user.is_authenticated:
            AssessmentHistory.objects.create(
                user=request.user,
                score=final_score,
                radar_data={
                    'labels': list(radar_data.keys()),
                    'values': list(radar_data.values())
                }
            )

        return render(request, 'predictor/result.html', {
            'score': final_score,
            'recommendations': recommendations,
            'age': age,
            'sex': int(sex),
            'res': res,
            'radar_labels': json.dumps(list(radar_data.keys()), ensure_ascii=False),
            'radar_values': json.dumps(list(radar_data.values())),
        })

    return render(request, 'predictor/predict.html')


def export_pdf_report(request):
    """تولید گزارش رسمی PDF همراه با نمودار راداری"""
    if request.method == 'POST':
        score = request.POST.get('score', '0')
        age = request.POST.get('age', '22')
        sex = request.POST.get('sex', '1')
        res = request.POST.get('res', '50')
        chart_image_data = request.POST.get('chart_image', None)
    else:
        score = request.GET.get('score', '0')
        age = request.GET.get('age', '22')
        sex = request.GET.get('sex', '1')
        res = request.GET.get('res', '50')
        chart_image_data = None

    sex_str = 'Male' if str(sex) == '1' else 'Female'

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="MindPulse_Wellbeing_Report.pdf"'

    doc = SimpleDocTemplate(response, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=30, bottomMargin=30)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#0d6efd'), alignment=1, spaceAfter=10)
    subtitle_style = ParagraphStyle('SubTitleStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#6c757d'), alignment=1, spaceAfter=15)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#198754'), spaceBefore=10, spaceAfter=8)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=9, leading=13, textColor=colors.HexColor('#212529'))

    # ۱. هدر سند
    story.append(Paragraph("MindPulse - Well-Being Assessment Report", title_style))
    story.append(Paragraph("Official AI-Based Psychological Evaluation Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0d6efd'), spaceAfter=15))

    # ۲. جدول خلاصه اطلاعات
    data = [
        [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value</b>", body_style)],
        [Paragraph("Age", body_style), Paragraph(str(age), body_style)],
        [Paragraph("Gender", body_style), Paragraph(sex_str, body_style)],
        [Paragraph("Resilience Score", body_style), Paragraph(f"{res} / 100", body_style)],
        [Paragraph("<b>Predicted Well-Being Score</b>", body_style), Paragraph(f"<b>{score} / 5.0</b>", body_style)]
    ]

    t = Table(data, colWidths=[200, 250])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9ecef')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#d1e7dd')),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

  
    if chart_image_data and ',' in chart_image_data:
        try:
            story.append(Paragraph("Psychological Profile (Radar Analysis)", heading_style))
            
           
            header, imgstr = chart_image_data.split(',', 1)
            img_data = base64.b64decode(imgstr)
            
            img_buffer = BytesIO(img_data)
            img_buffer.seek(0) 
            
            
            chart_img = RLImage(img_buffer, width=300, height=220)
            chart_img.hAlign = 'CENTER'
            story.append(chart_img)
            story.append(Spacer(1, 10))
            print("--- CHART ADDED TO PDF SUCCESSFULLY ---")
        except Exception as e:
            print(f"❌ ERROR RENDERING CHART IN PDF: {e}")
    else:
        print("⚠️ NO CHART DATA RECEIVED IN POST REQUEST!")

    # ۴. متدولوژی
    story.append(Paragraph("Model Methodology", heading_style))
    story.append(Paragraph("Calculated using a Hybrid Machine Learning Pipeline combining LightGBM Regressor & ElasticNet Regression.", body_style))
    story.append(Spacer(1, 15))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#adb5bd'), spaceAfter=10))
    story.append(Paragraph("Generated automatically by MindPulse Django Platform", ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.gray, alignment=1)))

    doc.build(story)
    return response


def compare_models(request):
    json_path = os.path.join(BASE_DIR, 'predictor', 'ml', 'model_comparison.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return render(request, 'predictor/compare.html', {'results': results})



@csrf_exempt
def chatbot_api(request):
    """ای پی آی چت‌بات هوشمند مبتنی بر Gemini AI"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            user_context = data.get('context', {}) # دریافت نمرات و پروفایل کاربر

            if not user_message:
                return JsonResponse({'status': 'error', 'message': 'پیام خالی است.'}, status=400)

            # فراخوانی کلاینت جدید Gemini
            client = genai.Client(api_key=settings.GEMINI_API_KEY)

            # ساخت دستورالعمل سیستمی برای هدایت لحن و نقش مدل
            system_prompt = (
                "شما یک دستیار هوشمند و همدل روان‌شناسی در سامانه MindPulse هستید. "
                "وظیفه شما تحلیل نتایج آزمون بهزیستی کاربر و ارائه راهکارهای عملی، علمی و انگیزشی برای بهبود سلامت روان است. "
                "لحن شما باید گرم، حرفه‌ای، محترمانه و کوتاه باشد. "
                "از ارائه تشخیص‌های پزشکی قطعی خودداری کنید و در صورت نیاز کاربر را به متخصص ارجاع دهید.\n\n"
                f"اطلاعات ارزیابی کاربر: {json.dumps(user_context, ensure_ascii=False)}"
            )

            # ارسال درخواست به مدل Gemini 2.5 Flash
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"{system_prompt}\n\nسوال کاربر: {user_message}"
            )

            return JsonResponse({
                'status': 'success',
                'reply': response.text
            })

        except Exception as e:
            print(f"❌ Gemini Chatbot Error: {e}")
            return JsonResponse({'status': 'error', 'message': 'خطا در برقراری ارتباط با مدل هوش مصنوعی.'}, status=500)

    return JsonResponse({'status': 'error', 'message': 'فقط درخواست POST پشتیبانی می‌شود.'}, status=405)


@login_required
def predict_view(request):
    if request.method == "POST":
        # ... کدهای مربوط به پیش‌بینی مدل ...
        score = calculated_score  # نمره به دست آمده
        radar_values = [...]       # مقادیر راداری

        
        AssessmentHistory.objects.create(
            user=request.user,
            score=score,
            radar_data={"labels": radar_labels, "values": radar_values}
        )
        # ...
        

@login_required
def history_view(request):
    # اصلاح order_order_by به order_by
    history = AssessmentHistory.objects.filter(user=request.user).order_by('created_at')
    
    dates = [h.created_at.strftime('%Y-%m-%d') for h in history]
    scores = [h.score for h in history]

    context = {
        'history': list(reversed(history)),  
        'dates': dates,
        'scores': scores,
    }
    return render(request, 'history.html', context)