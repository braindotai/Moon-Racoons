from rest_framework.response import Response
from rest_framework.decorators import api_view

from django.core.files.storage import default_storage
from django.conf import settings

import os
import cv2
import numpy as np
import base64
from PoseInference.core.detector import detect
import time

@api_view(['GET', 'POST'])
def InferenceView(request):
    if request.method == 'GET':
        return Response(
            {
                'asan': 'falana dhikana asan',
                'correct-posture': 'this would be an image'
            }
        )

    elif request.method == 'POST':
        # try:
        inference_file = request.FILES['face-image']
        inference_file_name = default_storage.save(inference_file.name, inference_file)
        inference_file_path = os.path.join(settings.MEDIA_ROOT, inference_file.name)

        inference_image = cv2.imread(inference_file_path)
        os.remove(inference_file_path)

        inference_output_image = detect(cv2_image = inference_image, verbose = True)

        _, jpg_encoded_image = cv2.imencode('.jpg', inference_output_image)
        output_image_bytes = jpg_encoded_image.tobytes()
        output_encoded_image = base64.b64encode(output_image_bytes)

        result = {'image_output': output_encoded_image}

        return Response({'result': 'success', 'message': 'inference successful', 'received': result})
        # except:
        #     return Response({'result': 'failure', 'message': 'uploaded image name is invalid'})