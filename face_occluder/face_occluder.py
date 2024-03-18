
import cv2
import numpy
import onnxruntime

class FACE_OCCLUDER:
   
    def __init__(self, model_path="face_occluder.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
        self.resolution = self.session.get_inputs()[0].shape[-2:]

    
    def create_occlusion_mask(self,crop_frame):

        prepare_frame = cv2.resize(crop_frame,(256,256))
        prepare_frame = numpy.expand_dims(prepare_frame, axis = 0).astype(numpy.float32) / 255
        prepare_frame = prepare_frame.transpose(0, 1, 2, 3)
        
        occlusion_mask = self.session.run(None,{self.session.get_inputs()[0].name: prepare_frame})[0][0]
        
        occlusion_mask = occlusion_mask.transpose(0, 1, 2).clip(0, 1).astype(numpy.float32)
        return occlusion_mask