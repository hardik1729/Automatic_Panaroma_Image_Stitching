import numpy as np
import cv2
import os
import sys
from matchers import matchers
import time
import stitch

def load_images_from_folder(folder):
	images = []
	for filename in sorted(os.listdir(folder)):
		img = cv2.resize(cv2.imread(os.path.join(folder,filename)),(480, 320))
		if img is not None:
			images.append(img)
	return images

def getFeatures(matcher_obj, arr):
	imageFeatureSet = []
	for each in arr:
		imageFeatureSet.append( matcher_obj.getSURFFeatures(each) )
	return imageFeatureSet

folder_name = 'temp'
img_arr = load_images_from_folder(folder_name)
print("number of images",len(img_arr))
matcher_obj = matchers(img_arr)
base=1
imageFeatureSet = getFeatures(matcher_obj, img_arr)
img_ind=matcher_obj.bestM_Matches(imageFeatureSet, base, 6)

size=[2048,2048,3]
result = np.zeros(size)
result_mask = np.zeros(size)
result = np.zeros(size)
xoffset=900
yoffset=900
result[xoffset:xoffset+img_arr[base].shape[0],yoffset:yoffset+img_arr[base].shape[1]]=img_arr[base]

single_mask=matcher_obj.gradient2d(img_arr[base])

result_mask[xoffset:xoffset+img_arr[base].shape[0],yoffset:yoffset+img_arr[base].shape[1],0]=single_mask
result_mask[xoffset:xoffset+img_arr[base].shape[0],yoffset:yoffset+img_arr[base].shape[1],1]=single_mask
result_mask[xoffset:xoffset+img_arr[base].shape[0],yoffset:yoffset+img_arr[base].shape[1],2]=single_mask

print(img_ind)
for i in img_ind:
	result,result_mask=stitch.combiner(result.astype('uint8'),result_mask,img_arr[i],xoffset,yoffset,size)
	cv2.imwrite("result1/result"+str(i)+".png",result)
	cv2.imwrite("result1/result_mask"+str(i)+".png",255*result_mask)
