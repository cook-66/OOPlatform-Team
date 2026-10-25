import os
import sys
import threading

from django.apps import AppConfig


class PlatformWebConfig(AppConfig):
    name = 'platform_web'

    def ready(self):
        """runserver 启动时后台预热 YOLO 模型，避免用户第一次点击还要等初始化。"""
        if 'runserver' not in sys.argv:
            return
        # autoreload 会起两个进程，只在真正提供服务的那个里预热；--noreload 时只有一个进程
        if os.environ.get('RUN_MAIN') != 'true' and '--noreload' not in sys.argv:
            return
        from .views import warmup_yolo_model
        threading.Thread(target=warmup_yolo_model, name='yolo-warmup', daemon=True).start()
