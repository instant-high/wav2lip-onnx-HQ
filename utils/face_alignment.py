import cv2
import numpy as np


def align_crop(img, landmark, size):
    template_ffhq = np.array(
	[
		[192.98138, 239.94708],
		[318.90277, 240.19366],
		[256.63416, 314.01935],
		[201.26117, 371.41043],
		[313.08905, 371.15118]
	])
    
    template_ffhq *= (512 / size)
    matrix = cv2.estimateAffinePartial2D(landmark, template_ffhq, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
    warped = cv2.warpAffine(img, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)
    return warped, matrix

def get_cropped_head(img, landmark, scale=1.4, size=512):
    center = np.mean(landmark, axis=0)
    landmark = center + (landmark - center) * scale
    return align_crop(img, landmark, size)
    
    
# --------------------------------------------------
    
def align_crop_256(img, landmark, size):
    template_ffhq = np.array(
	[
		[192.98138, 239.94708],
		[318.90277, 240.19366],
		[256.63416, 314.01935],
		[201.26117, 371.41043],
		[313.08905, 371.15118]
	])
    
    template_ffhq = template_ffhq /2
    template_ffhq *= (256 / size)
    matrix = cv2.estimateAffinePartial2D(landmark, template_ffhq, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
    warped = cv2.warpAffine(img, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)
    return warped, matrix


def get_cropped_head_256(img, landmark, scale=1.4, size=512):
    center = np.mean(landmark, axis=0)
    landmark = center + (landmark - center) * scale
    return align_crop_256(img, landmark, size)

def get_cropped(img, landmark, scale=1.4, size=512, bbox_expansion_factor=3):
    # Scale landmarks around the center
    center = np.mean(landmark, axis=0)
    scaled_landmark = center + (landmark - center) * scale
    
    # Calculate the bounding box
    min_coords = np.min(scaled_landmark, axis=0)
    max_coords = np.max(scaled_landmark, axis=0)
    
    width, height = max_coords - min_coords
    max_dim = max(width, height)
    
    # Expand the bounding box by the specified factor
    expanded_dim = max_dim * bbox_expansion_factor
    
    # Calculate the expanded bounding box coordinates
    center_x, center_y = (min_coords + max_coords) / 2
    half_expanded_dim = expanded_dim / 2
    min_x = max(int(center_x - half_expanded_dim), 0)
    min_y = max(int(center_y - half_expanded_dim), 0)
    max_x = min(int(center_x + half_expanded_dim), img.shape[1])
    max_y = min(int(center_y + half_expanded_dim), img.shape[0])
    
    # Crop and resize the image
    cropped_img = img[min_y:max_y, min_x:max_x]
    cropped_img_resized = cv2.resize(cropped_img, (size, size))
    
    return cropped_img_resized
           
