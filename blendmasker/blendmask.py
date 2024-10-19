import numpy as np
import onnxruntime

class BLENDMASK:
    def __init__(self, model_path="blendswap_256.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)

    def mask(self, target_face):

        target_face = target_face.astype(np.float32)
        target_face = target_face[..., ::-1]
        target_face = target_face.transpose((2, 0, 1))
        target_face = target_face /255.0
        target_face = np.expand_dims(target_face, axis=0).astype(np.float32)
        
        res = self.session.run(None, {(self.session.get_inputs()[0].name):target_face})[0]
        
        res = res.squeeze()
        res = res * 255.0
        res = res.astype(np.uint8)
        res = np.stack([res] * 3, axis=-1)
         
        return res