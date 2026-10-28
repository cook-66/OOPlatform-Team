from django.contrib import admin
from django.urls import include, path, re_path

from platform_web.views import serve_media

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include("platform_web.urls")),
    re_path(r'^media/(?P<path>.*)$', serve_media),
]
