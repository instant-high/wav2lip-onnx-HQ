import cv2
import onnxruntime
import numpy as np
import imutils
from skimage.transform import resize

class SEGMENTATION_MODULE:
    def __init__(self, model_path="vox-5segments.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)

        
    def mask(self, face, FACE_MASK_REGIONS):
        face = resize(face.astype(np.uint8), (256, 256))[..., :3]
        face = face.transpose((2, 0, 1))
        face = np.expand_dims(face, axis=0)
        face = face.astype(np.float32)[:1,::-1]
        
        region_mask = self.session.run(None, {(self.session.get_inputs()[0].name):face})[0][0]
        
        region_mask = np.isin(region_mask.argmax(0), FACE_MASK_REGIONS)

        region_mask = region_mask.astype(np.uint8) *255
        region_mask = cv2.GaussianBlur(region_mask,(5,5),cv2.BORDER_DEFAULT)
        region_mask = cv2.resize(region_mask, (256,256))
      
        return region_mask
        
