from django.contrib import admin

from .models import InferenceRecord


@admin.register(InferenceRecord)
class InferenceRecordAdmin(admin.ModelAdmin):
    list_display = ["id", "detected_classes", "confidence", "created_at"]
    list_filter = ["detected_classes"]
    search_fields = ["detected_classes"]
