import subprocess
import platform
import numpy as np
import cv2, os, sys, argparse, audio, shutil

from os import listdir, path
from tqdm import tqdm
from PIL import Image

import onnxruntime
onnxruntime.set_default_logger_severity(3)

from insightface_func.crop_single import Face_detect_crop


parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', required=True)
parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--final_audio', type=str, help='Filepath of video/audio file to use as final audio source')
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', default='results/result_voice.mp4')

parser.add_argument('--static', default=False, action='store_true', help='If True, then use only first video frame for inference')
parser.add_argument('--pingpong', default=False, action='store_true',help='pingpong loop if audio is longer than video')

parser.add_argument("--cut_in", type=int, default=0, help="Frame to start inference")
parser.add_argument("--cut_out", type=int, default=0, help="Frame to end inference")

parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', default=25., required=False)
parser.add_argument('--resize_factor', default=1, type=int, help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument("--enhancer", default='none', choices=['none', 'gpen', 'gfpgan', 'codeformer', 'restoreformer'])
parser.add_argument('--blending', default=7, type=float, help='Amount of face enhancement blending 1 - 10')

parser.add_argument("--sharpen", default=False, action="store_true", help="Slightly sharpen swapped face")

parser.add_argument('--preview', default=False, action='store_true', help='Preview during inference')

parser.add_argument("--segmentation", action="store_true", help="Use face_segmentation mask")
parser.add_argument("--seg_index", default="1,2,5", type=lambda x: list(map(int, x.split(','))),help='index of enhanced face parts') # 1,2,5

parser.add_argument("--face_occluder", action="store_true", help="Use occluder face mask")

parser.add_argument('--pads', type=int, default=0, help='Padding top, bottom to adjust best mouth position, move crop up/down') # pos value mov synced mouth up

parser.add_argument('--hq_output', default=False, action='store_true',help='HQ output')

#Removed
#parser.add_argument('--face_det_batch_size', type=int, help='Batch size for face detection', default=16)
#parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=1)
#parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
#parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.''Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
#parser.add_argument('--rotate', default=False, action='store_true',help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.''Use if you get a flipped result, despite feeding a normal looking video')
#parser.add_argument('--nosmooth', default=False, action='store_true',help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()

args.img_size = 96
mel_step_size = 16

padY = args.pads

device = 'cpu'
if onnxruntime.get_device() == 'GPU': device = 'cuda'


if args.enhancer == 'gpen':
	from enhancers.GPEN.GPEN import GPEN
	gpen256 = GPEN(model_path="enhancers/GPEN/GPEN-BFR-256-sim.onnx", device=device)
	#gpen256 = GPEN(model_path="enhancers/GPEN/GPEN-BFR-512.onnx", device=device)

if args.enhancer == 'codeformer':
		from enhancers.Codeformer.Codeformer import CodeFormer
		codeformer = CodeFormer(model_path="enhancers/Codeformer/codeformerfixed.onnx", device=device)

if args.enhancer == 'restoreformer':
		from enhancers.restoreformer.restoreformer16 import RestoreFormer
		restoreformer = RestoreFormer(model_path="enhancers/restoreformer/restoreformer16.onnx", device=device)
		
if args.enhancer == 'gfpgan':
		from enhancers.GFPGAN.GFPGAN import GFPGAN
		gfpgan = GFPGAN(model_path="enhancers/GFPGAN/GFPGANv1.4.onnx", device=device)
		   
if args.segmentation:
    from seg_mask.seg_mask import SEGMENTATION_MODULE
    seg_module = SEGMENTATION_MODULE(model_path="seg_mask/vox-5segments.onnx", device=device)

if args.face_occluder:
		from face_occluder.face_occluder import FACE_OCCLUDER
		occluder = FACE_OCCLUDER(model_path="face_occluder/face_occluder.onnx", device=device)
        		    
if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static: args.static = True



def load_model(device):
	model_path = args.checkpoint_path
	session_options = onnxruntime.SessionOptions()
	session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
	providers = ["CPUExecutionProvider"]
	if device == 'cuda':
		providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
		
	session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)	
	
	return session
		

def face_detect(images):

	detector = Face_detect_crop(name='antelope', root='./insightface_func/models')
	detector.prepare(ctx_id= 0, det_thresh=0.3, det_size=(320,320),mode='ffhq')
	
	os.system('cls')
	print ("Detecting face and generating data...")
					
	crop_size = 256

	sub_faces = []
	crop_faces = []
	matrix = []
	face_error = []
				
	for i in tqdm(range(0, len(images))):

		try:	
			crop_face, M = detector.get(images[i],crop_size)
			
			sub_face = crop_face[65-(padY):241-(padY),62:194] # 176x132 /  ffhq 1.[60:236,62:194] 2.[65:241,62:194] yy xx
			sub_face = cv2.resize(sub_face, (96,96))  
			
			# for demo video
			#if i == 0:
			#	crop_copy = crop_face.copy()
			#	x, y, w, h = 62, 65 - padY, 132, 176
			#	cv2.rectangle(crop_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)					
			#	cv2.imshow("Aligned face",crop_copy)
			#	cv2.waitKey()
			#	cv2.destroyAllWindows()

			sub_faces.append(sub_face)		
			crop_faces.append(crop_face)
			matrix.append(M)

			no_face = 0
			
		except:
			if i == 0:
				crop_face = np.zeros((256,256), dtype=np.uint8)
				crop_face = cv2.cvtColor(crop_face, cv2.COLOR_GRAY2RGB)/255
				sub_face = crop_face[65-(padY):241-(padY),62:194]
				sub_face = cv2.resize(sub_face, (96,96))
				M = np.float32([[1,2,3],[1,2,3]])
								
			sub_faces.append(sub_face)		
			crop_faces.append(crop_face)
			matrix.append(M)
			
			no_face = -1
			
		face_error.append(no_face)
		
	return crop_faces, sub_faces, matrix, face_error 

def datagen(frames, mels):
	
	img_batch, mel_batch, frame_batch = [], [], []

	for i, m in enumerate(mels):
	
		idx = 0 if args.static else i%len(frames)

		frame_to_save = frames[idx].copy()
		frame_batch.append(frame_to_save)
			
		img_batch.append(frames[idx])
		mel_batch.append(m)

		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0
		
		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch
		img_batch, mel_batch, frame_batch = [], [], []
    

def main():
	if args.hq_output:
		if not os.path.exists('hq_temp'):
			os.mkdir('hq_temp')
	
	blend = args.blending/10
 
	static_face_mask = np.zeros((224,224), dtype=np.uint8)
	static_face_mask = cv2.ellipse(static_face_mask, (112,162), (62,54),0,0,360,(255,255,255), -1)
	static_face_mask = cv2.ellipse(static_face_mask, (112,122), (46,23),0,0,360,(0,0,0), -1)
	static_face_mask = cv2.resize(static_face_mask,(256,256))
	
	static_face_mask = cv2.rectangle(static_face_mask, (0,236), (256,256),(0,0,0), -1) # not(0,242)
	
	static_face_mask = cv2.cvtColor(static_face_mask, cv2.COLOR_GRAY2RGB)/255
	static_face_mask = cv2.GaussianBlur(static_face_mask,(29,29),cv2.BORDER_DEFAULT)

	sub_face_mask = np.zeros((256,256), dtype=np.uint8)

	sub_face_mask = cv2.rectangle(sub_face_mask, (66,69), (190,240),(255,255,255), -1)  #[65:241,62:194]  ###(66,69), (190,235)
	sub_face_mask = cv2.GaussianBlur(sub_face_mask.astype(np.uint8),(19,19),cv2.BORDER_DEFAULT)
	sub_face_mask = cv2.cvtColor(sub_face_mask, cv2.COLOR_GRAY2RGB)
	sub_face_mask = sub_face_mask/255
	
	#cv2.imshow("Static mask",static_face_mask)				
	#cv2.imshow("Sub mask",sub_face_mask)
	#cv2.waitKey()
		
	im = cv2.imread(args.face)

	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')
	
	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		orig_frame = cv2.imread(args.face)
		orig_frame = cv2.resize(orig_frame, (orig_frame.shape[1]//args.resize_factor, orig_frame.shape[0]//args.resize_factor))	
		orig_frames = [orig_frame]
		fps = args.fps

#  crop image:
		h, w = orig_frame.shape[:-1]
		roi = cv2.selectROI("Select region of target face", orig_frame, showCrosshair=False)
		if roi == (0,0,0,0):roi = (0,0,w,h)
		cropped_roi = orig_frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
		cv2.destroyAllWindows()
		full_frames = [cropped_roi]
				
	else:
		video_stream = cv2.VideoCapture(args.face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)
		video_stream.set(1,args.cut_in)

		print('Reading video frames...')
		
#   cut to input/putput position:

		if args.cut_out == 0:
			args.cut_out = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
			
		duration = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT)) - args.cut_in
		new_duration = args.cut_out - args.cut_in
		
		if args.static:
			new_duration = 1
#
		
		video_stream.set(1,args.cut_in)

			
#   read frames and crop roi:

		full_frames = []
		orig_frames = []
		
		for l in range(new_duration):
			still_reading, frame = video_stream.read()
			
			if not still_reading:
				video_stream.release()
				break
				
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))			

#  crop first frame:
			if l == 0:
				h, w = frame.shape[:-1]
				roi = cv2.selectROI("Select region of target face", frame, showCrosshair=False)
				if roi == (0,0,0,0):roi = (0,0,w,h)
				cropped_roi = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
				cv2.destroyAllWindows()

#     crop all frames:
			cropped_roi = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
			full_frames.append(cropped_roi)
			orig_frames.append(frame)

	#print(len(full_frames))
	#print(len(orig_frames))
	#input("1")			

# memory usage raw video
	memory_usage_bytes = sum(frame.nbytes for frame in full_frames)
	#memory_usage_kb = memory_usage_bytes / 1024
	memory_usage_mb = memory_usage_bytes / (1024**2)
	
	print ("Number of frames used for inference: " + str(len(full_frames)) + " / ~ " + str(int(memory_usage_mb)) + " mb memory usage")
	
	
	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'

	wav = audio.load_wav(args.audio, 16000)
	mel = audio.melspectrogram(wav)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

# new face detection:	
	aligned_faces, sub_faces, matrix, no_face = face_detect(full_frames)

	if args.pingpong:
		orig_frames = orig_frames + orig_frames[::-1]
		full_frames = full_frames + full_frames[::-1]
		aligned_faces = aligned_faces + aligned_faces[::-1]
		sub_faces = sub_faces + sub_faces[::-1]
		matrix = matrix + matrix[::-1]
		no_face = no_face + no_face[::-1]

# new datagen:					
	gen = datagen(sub_faces.copy(), mel_chunks)
	
	fc = 0

	model = load_model(device)

	frame_h, frame_w = full_frames[0].shape[:-1]
	orig_h, orig_w = orig_frames[0].shape[:-1]
	out = cv2.VideoWriter('temp/temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h))
				
	os.system('cls')
	print('Running on ' + onnxruntime.get_device())
	print ('Checkpoint: ' + args.checkpoint_path)
	print ('Resize factor: ' + str(args.resize_factor))
	if args.pingpong: print ('Use pingpong')
	if args.enhancer != 'none': print ('Use ' + args.enhancer)
	if args.segmentation: print ('Use segmentation mask')
	if args.face_occluder: print ('Use face occluder')
	print ('')

	
	for i, (img_batch, mel_batch, frames) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)))))):
					
		f_len = len(full_frames)
		if fc == (len(full_frames)):
			fc = 0
			
		face_err = no_face[fc]
		
		img_batch = img_batch.transpose((0, 3, 1, 2)).astype(np.float32)
		mel_batch = mel_batch.transpose((0, 3, 1, 2)).astype(np.float32)		

		pred = model.run(None,{'mel_spectrogram':mel_batch, 'video_frames':img_batch})[0][0]
		pred = pred.transpose(1, 2, 0)*255
		pred = pred.astype(np.uint8)
		pred = pred.reshape((1, 96, 96, 3))		
		
		mat = matrix[fc]
		mat_rev = cv2.invertAffineTransform(mat)

		aligned_face = aligned_faces[fc]
		aligned_face_orig = aligned_face.copy()
		p_aligned = aligned_face.copy()
		
		full_frame = full_frames[fc]
#
		final = orig_frames[fc]
#	
		for p, f in zip(pred, frames):			
				
			if not args.static: fc = fc + 1

			p = cv2.resize(p,(132,176))
			
			#cv2.imshow("P",p)
			#input(p.shape)
			#aligned_face[65:241,62:194] = p  # [60:236,62:194]
			
			###p_aligned[65-(padY*4):241,62:194] = p
			p_aligned[65-(padY):241-(padY),62:194] = p
			
			aligned_face = (sub_face_mask * p_aligned + (1 - sub_face_mask) * aligned_face_orig).astype(np.uint8)

			#cv2.imshow("Result",aligned_face)
			#cv2.imshow("Orig.",aligned_face_orig)
			#cv2.waitKey()
			
			if face_err != 0:
				res = full_frame
				face_err = 0
				
			else:
#
				if args.enhancer == 'gpen':
					#aligned_face = cv2.resize(aligned_face,(256,256))
					aligned_face_enhanced = gpen256.enhance(aligned_face)
					aligned_face_enhanced = cv2.resize(aligned_face_enhanced,(256,256))
					aligned_face = cv2.addWeighted(aligned_face_enhanced.astype(np.float32),blend, aligned_face.astype(np.float32), 1.-blend, 0.0)

				if args.enhancer == 'codeformer':                
					#aligned_face = cv2.resize(aligned_face,(512,512))
					aligned_face_enhanced = codeformer.enhance(aligned_face,0.6)
					aligned_face_enhanced = cv2.resize(aligned_face_enhanced,(256,256))
					aligned_face = cv2.addWeighted(aligned_face_enhanced.astype(np.float32),blend, aligned_face.astype(np.float32), 1.-blend, 0.0)
					
				if args.enhancer == 'restoreformer':                
					#aligned_face = cv2.resize(aligned_face,(512,512))
					aligned_face_enhanced = restoreformer.enhance(aligned_face)
					aligned_face_enhanced = cv2.resize(aligned_face_enhanced,(256,256))
					aligned_face = cv2.addWeighted(aligned_face_enhanced.astype(np.float32),blend, aligned_face.astype(np.float32), 1.-blend, 0.0)
					
				if args.enhancer == 'gfpgan':                
					#aligned_face = cv2.resize(aligned_face,(512,512))
					aligned_face_enhanced = gfpgan.enhance(aligned_face)
					aligned_face_enhanced = cv2.resize(aligned_face_enhanced,(256,256))
					aligned_face = cv2.addWeighted(aligned_face_enhanced.astype(np.float32),blend, aligned_face.astype(np.float32), 1.-blend, 0.0)
					
#
				if args.segmentation:
					seg_mask = seg_module.mask(aligned_face, args.seg_index)
					seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
					seg_mask = cv2.rectangle(seg_mask,(10,252),(246,256),(0,0,0), -1)
					seg_mask = cv2.rectangle(seg_mask, (0,236), (256,256),(0,0,0), -1)
					seg_mask = cv2.GaussianBlur(seg_mask,(19,9),cv2.BORDER_DEFAULT)
					seg_mask = seg_mask /255
					mask = cv2.warpAffine(seg_mask, mat_rev,(frame_w, frame_h))
					
				if args.face_occluder:
					seg_mask = occluder.create_occlusion_mask(aligned_face)
					seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
					seg_mask = cv2.rectangle(seg_mask, (5,5), (251,251), (0, 0, 0), 10)
					seg_mask = cv2.GaussianBlur(seg_mask,(19,19),cv2.BORDER_DEFAULT)
				
				if not args.segmentation and not args.face_occluder:
					mask = cv2.warpAffine(static_face_mask, mat_rev,(frame_w, frame_h))		
#	


				if args.sharpen:
					smoothed = cv2.GaussianBlur(aligned_face, (9, 9), 10)
					aligned_face = cv2.addWeighted(aligned_face, 1.5, smoothed, -0.5, 0)
					aligned_face = np.clip(aligned_face, 0, 255).astype(np.uint8)
					

				dealigned_face =  cv2.warpAffine(aligned_face, mat_rev, (frame_w, frame_h))
				#cv2.imshow("msk",mask)
				#cv2.waitKey(1)
				#mask = cv2.warpAffine(static_face_mask, mat_rev,(frame_w, frame_h))
				
				res = (mask * dealigned_face + (1 - mask) * full_frame).astype(np.uint8)

#   insert cropped region:
		final[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] = res

		if args.hq_output:
			cv2.imwrite(os.path.join('./hq_temp', '{:0>7d}.png'.format(i)), final)
		else:	
			out.write(final)

		if args.preview:
			cv2.imshow("Result - press ESC to stop and save",final)
			k = cv2.waitKey(1)
			if k == 27:
				cv2.destroyAllWindows()
				out.release()
				break

			if k == ord('s'):
				if args.sharpen == False:
					args.sharpen = True
				else:
					args.sharpen = False
				print ('')    
				print ("Sharpen = " + str(args.sharpen))
                
		#if fc == (len(full_frames)): break
						
	out.release()

	if args.final_audio:
		if args.hq_output:	
			command = 'ffmpeg.exe -y -i ' + '"' + args.final_audio + '"' + ' -r ' + str(fps) + ' -f image2 -i ' + '"' + './hq_temp/' + '%07d.png' + '"' + ' -shortest -vcodec libx264 -pix_fmt yuv420p -preset slow -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -strict -2 -q:v 1 ' + '"' + args.outfile + '"'						
		else:
			command = 'ffmpeg.exe -y -i ' + '"' + args.final_audio + '"' + ' -i ' + 'temp/temp.mp4' + ' -shortest -vcodec copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -strict -2 -q:v 1 ' + '"' + args.outfile + '"'
		subprocess.call(command, shell=platform.system() != 'Windows')
		
		if os.path.exists('temp/temp.mp4'):
			os.remove('temp/temp.mp4')
		if  os.path.exists('hq_temp'):
			shutil.rmtree('hq_temp')		
	else:
		if args.hq_output:
		  command = 'ffmpeg.exe -y -i ' + '"' + args.audio + '"' + ' -r ' + str(fps) + ' -f image2 -i ' + '"' + './hq_temp/' + '%07d.png' + '"' + ' -shortest -vcodec libx264 -pix_fmt yuv420p -preset slow -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -strict -2 -q:v 1 ' + '"' + args.outfile + '"'						
		else:						
			command = 'ffmpeg.exe -y -i ' + '"' + args.audio + '"' + ' -i ' + 'temp/temp.mp4' + ' -shortest -vcodec copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -strict -2 -q:v 1 ' + '"' + args.outfile + '"'

		subprocess.call(command, shell=platform.system() != 'Windows')
		
		if os.path.exists('temp/temp.mp4'):
			os.remove('temp/temp.mp4')
		if  os.path.exists('hq_temp'):
			shutil.rmtree('hq_temp')	

if __name__ == '__main__':
	main()
