# -*- coding: utf-8 -*-
"""
Face Anti-Spoofing Detection Module
Detects real vs fake faces using deep learning models
"""

import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F


from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}


class Detection:
    def __init__(self, base_dir=None):
        """
        Initialize face detection with RetinaFace model
        
        Args:
            base_dir: Base directory of face_security module. If None, will auto-detect.
        """
        if base_dir is None:
            # Auto-detect base directory (face_security folder)
            current_file = os.path.abspath(__file__)
            # Go up from src/anti_spoof_predict.py to face_security/
            base_dir = os.path.dirname(os.path.dirname(current_file))
        
        detection_model_dir = os.path.join(base_dir, "resources", "detection_model")
        caffemodel = os.path.join(detection_model_dir, "Widerface-RetinaFace.caffemodel")
        deploy = os.path.join(detection_model_dir, "deploy.prototxt")
        
        if not os.path.exists(caffemodel) or not os.path.exists(deploy):
            raise FileNotFoundError(
                f"Detection model files not found in {detection_model_dir}. "
                f"Please ensure Widerface-RetinaFace.caffemodel and deploy.prototxt are present."
            )
        
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6
        self.base_dir = base_dir

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox


class AntiSpoofPredict(Detection):
    def __init__(self, device_id, base_dir=None):
        """
        Initialize anti-spoofing predictor
        
        Args:
            device_id: CUDA device ID (or 0 for CPU)
            base_dir: Base directory of face_security module. If None, will auto-detect.
        """
        super(AntiSpoofPredict, self).__init__(base_dir=base_dir)
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_path):
        """
        Load anti-spoofing model from file
        
        Args:
            model_path: Path to model file (.pth). Can be relative or absolute.
        """
        # If model_path is relative, resolve it relative to base_dir
        if not os.path.isabs(model_path):
            model_path = os.path.join(self.base_dir, "resources", "anti_spoof_models", model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input,)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def predict(self, img, model_path):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self._load_model(model_path)
        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result).cpu().numpy()
        return result



