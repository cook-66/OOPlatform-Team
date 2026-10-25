from django.db import models


class InferenceRecord(models.Model):
    original_img = models.ImageField(upload_to="original/", verbose_name="原图")
    result_img = models.ImageField(upload_to="result/", verbose_name="结果图", blank=True)
    detected_classes = models.CharField(max_length=200, verbose_name="检测类别", blank=True)
    confidence = models.FloatField(verbose_name="置信度", default=0.0)
    llm_explanation = models.TextField(verbose_name="智能解释", blank=True)
    llm_cost = models.FloatField(default=0.0, verbose_name="预估花费")
    llm_tokens = models.IntegerField(default=0, verbose_name="Token 消耗")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="检测时间")

    class Meta:
        verbose_name = "推理记录"
        verbose_name_plural = "推理记录"
