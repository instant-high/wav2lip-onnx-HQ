# wav2lip-onnx-HQ
Update 2024.10.19

  - Replaced insightface with retinaface detection/alignment for easier installation
  - Replaced seg-mask with faster blendmasker
  - Added free cropping of final result video
  - Added specific target face selection from first frame

.

Just another Wav2Lip HQ local installation, fully running on Torch to ONNX converted models for:
- face-detection
- face-recognition
- face-alignment
- face-parsing
- face-enhancement
- wav2lip inference.

.

Can be run on CPU or Nvidia GPU

I've made some modifications such as:
* New face-detection and face-alignment code. (working for ~ +- 60º head tilt)
* Four different face enhancers available, adjustable enhancement level .
* Choose pingpong loop instead of original loop function.
* Set cut-in/cut-out position to create the loop or cut longer video.
* Cut-in position = used frame if static is selected.
* Select the target face area, not  a real face recognition, which also makes inference faster.
* Use two audio files, eg. vocal for driving and full music mix for final output.
* This version does not crash if no face is detected, it just continues ...

Type --help for all commandline parameters

.
 
Model download - https://drive.google.com/drive/folders/1BGl9bmMtlGEMx_wwKufJrZChFyqjnlsQ?usp=sharing


Original wav2lip - https://github.com/Rudrabha/Wav2Lip

Face enhancers taken from -  https://github.com/harisreedhar/Face-Upscalers-ONNX

Face detection taken from - https://github.com/neuralchen/SimSwap

Face occluder taken from - https://github.com/facefusion/facefusion-assets/releases

Blendmasker extracted from - https://github.com/mapooon/BlendFace during onnx conversion

Face recognition for specifc face taken from - https://github.com/jahongir7174/FaceID

.

.


