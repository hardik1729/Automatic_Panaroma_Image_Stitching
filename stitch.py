import numpy as np
import cv2
import sys
from matchers import matchers
import multi_band_blending
import time

def gradient2d(i1):
	return np.outer(gradient1d(i1.shape[0]), gradient1d(i1.shape[1]))

def gradient1d(x):
	d = int(x/2)
	arr = np.empty(x)
	tmp1=np.arange(d+1)/d
	tmp2=1-np.arange(x-d)/(x-1-d)
	return np.append(tmp1, tmp2[1:])

def combiner(a,mask,b,offsetx,offsety,dsize):
	dsize=(dsize[0],dsize[1])
	print(a.shape,b.shape)
	H = matchers().match(a, b, 1)
	print("Homography is : ", H)
	xh = H.copy()
	offset=H.copy()
	offset.fill(0)
	offset=offset+np.array([[1,0,offsetx],[0,1,offsety],[0,0,1]])

	print("Homography :", xh)

	img2 = cv2.warpPerspective(b, xh, dsize)
	cv2.imwrite('2.png', img2)

	img1=a.copy()
	#img1[offsety:a.shape[0]+offsety, offsetx:a.shape[1]+offsetx] = a
	cv2.imwrite('1.png', img1)

	#-------------------------------------

	grad_b = matchers().gradient2d(b)

	#multi_grad_a=np.zeros(a.shape)
	multi_grad_b=np.zeros(b.shape)

	multi_grad_a=mask.copy()
	
	multi_grad_b[:,:,0]=grad_b
	multi_grad_b[:,:,1]=grad_b
	multi_grad_b[:,:,2]=grad_b

	mask1 = multi_grad_a.copy()
	cv2.imwrite('warp-mask1.png', 255*mask1)

	mask2 = cv2.warpPerspective(multi_grad_b, xh, dsize)
	cv2.imwrite('warp-mask2.png', 255*mask2)

	result_mask=mask1.copy()
	result_mask[np.where(mask1<mask2)]=mask2[np.where(mask1<mask2)]

	img2=img2.astype(float)
	result = multi_band_blending.multi_band_blending(img1, img2, mask1, mask2)
	cv2.imwrite('result.png', result)
	return result,result_mask