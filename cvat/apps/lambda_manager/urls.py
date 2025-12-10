# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from django.urls import include, path
from rest_framework import routers

from . import views
from .views_sam3 import SAM3ViewSet
from .views_yoloe import YOLOEVisualPromptViewSet

router = routers.DefaultRouter(trailing_slash=False)
# https://github.com/encode/django-rest-framework/issues/6645
# I want to "call" my functions. To do that need to map my call method to
# POST (like get HTTP method is mapped to list(...)). One way is to implement
# own CustomRouter. But it is simpler just patch the router instance here.
router.routes[2].mapping.update({"post": "call"})
router.register("functions", views.FunctionViewSet, basename="lambda_function")
router.register("requests", views.RequestViewSet, basename="lambda_request")
router.register("yoloe", YOLOEVisualPromptViewSet, basename="lambda_yoloe")
router.register("sam3", SAM3ViewSet, basename="lambda_sam3")

# GET  /api/lambda/functions - get list of functions
# GET  /api/lambda/functions/<int:fid> - get information about the function
# POST /api/lambda/requests - call a function
# { "function": "<id>", "mode": "online|offline", "job": "<jid>", "frame": "<n>",
#   "points": [...], }
# GET  /api/lambda/requests - get list of requests
# GET  /api/lambda/requests/<int:rid> - get status of the request
# DEL  /api/lambda/requests/<int:rid> - cancel a request (don't delete)
#
# YOLOE Visual Prompt endpoints:
# GET  /api/lambda/yoloe/annotated-frames?job_id=N - get frames with annotations
# POST /api/lambda/yoloe/generate-vpe - generate VPE from references
# POST /api/lambda/yoloe/predict - run detection on frames
# GET  /api/lambda/yoloe/status?job_id=N - get VPE cache status
# POST /api/lambda/yoloe/clear - clear VPE cache
# POST /api/lambda/yoloe/apply - apply predictions as annotations
#
# SAM3 endpoints:
# POST /api/lambda/sam3/segment-text - segment using text prompt
# POST /api/lambda/sam3/segment-points - segment using click points
# POST /api/lambda/sam3/segment-combined - segment using text + points
# POST /api/lambda/sam3/detect - detect all instances matching text
# POST /api/lambda/sam3/track/init - initialize video tracking
# POST /api/lambda/sam3/track/frame - track to next frame
# GET  /api/lambda/sam3/track/status - get tracking session status
# POST /api/lambda/sam3/track/clear - clear tracking session
# GET  /api/lambda/sam3/cache/status - get embeddings cache status
# POST /api/lambda/sam3/cache/clear - clear embeddings cache
# POST /api/lambda/sam3/apply - apply detections as annotations
urlpatterns = [path("api/lambda/", include(router.urls))]
