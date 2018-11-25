import cv2
import numpy as np 
import matplotlib.pyplot as plt

class matchers:
	def __init__(self, img_arr=None):
		self.surf = cv2.xfeatures2d.SURF_create()
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=0, trees=5)
		search_params = dict(checks=50)
		self.flann = cv2.FlannBasedMatcher(index_params, search_params)
		self.img_arr = img_arr 
	
	def gradient2d(self, i1):
		return np.outer(self.gradient1d(i1.shape[0]), self.gradient1d(i1.shape[1]))

	def gradient1d(self, x):
		d = int(x/2)
		arr = np.empty(x)
		tmp1=np.arange(d+1)/d
		tmp2=1-np.arange(x-d)/(x-1-d)
		return np.append(tmp1, tmp2[1:])

	def bestM_Matches(self, imageFeatureSet, N, M):
		featureMatched_arr = []
		for i,imageSet in enumerate(imageFeatureSet):
			if(i==N):
				continue
			
			matches = self.flann.knnMatch(
				imageSet['des'],
				imageFeatureSet[N]['des'],
				k=2
				)

			good = []
			for (m, n) in matches:
				if m.distance < 0.7*n.distance:
					good.append((m.trainIdx, m.queryIdx))

			if len(good) > 4:
				featureMatched_arr.append([i,good])
				print(i, len(good))

		def order(arr_element):
			return len(arr_element[1])

		featureMatched_arr.sort(key=order, reverse=True)
		small_featureMatched_arr = featureMatched_arr[:M]

		def total_features_in_intersection(a, b, xh):
			dsize = (a.shape[1],a.shape[0])
			b_mask = np.ones(b.shape)*255
			intersection = cv2.warpPerspective(b_mask, xh, dsize)
			nf = self.featuresInMask(a,intersection[:,:,0])
			return nf

		final_arr_index = []
		for [i,good] in small_featureMatched_arr:
			pointsCurrent = imageFeatureSet[i]['kp'] #query
			pointsPrevious = imageFeatureSet[N]['kp'] #train

			#KeyPoint.pt - point2D of the KeyPoint
			matchedPointsCurrent = np.float32(
				[pointsCurrent[index].pt for (__, index) in good]
			)
			matchedPointsPrev = np.float32(
				[pointsPrevious[index].pt for (index, __) in good]
				)

			#returns the 3x3 perspective transformation H on the source[xi,yi,1] to reach the destination planes[xf,yf,1]
			H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
			ni = len(good)
			nf = total_features_in_intersection(self.img_arr[N],self.img_arr[i],H)

			alpha=8
			beta=0.1
			# if (ni>(alpha+beta*nf)):
			final_arr_index.append(i)
			print(i, ni>(alpha+beta*nf))

		return final_arr_index

	def match(self, i1, i2, direction=None, pltflag=0):
		imageSet1 = self.getSURFFeatures(i1) #dict {kp: [7391 list of keypoint class objects], des: 7391x64 np array} 7391 is the number of features, that can vary
		imageSet2 = self.getSURFFeatures(i2)
		# print("Direction : ", direction)

		#returns a list of lists - [no of kp of imageSet2, k] where k is the k closest des in the train vector
		#train - imageSet1, query - imageSet2 ------ Finds the k best matches for each descriptor from a query set
		matches = self.flann.knnMatch(
			imageSet2['des'],
			imageSet1['des'],
			k=2
			)
		print(len(matches))

		good = []
		goodDMatch=[]
		#ratio test - Lowe paper - If the closest match is ratio closer than the second closest one, then the match is correct.
		#trainIdx - kp of train des, queryIdx - kp of query des 
		for i , (m, n) in enumerate(matches):
			if m.distance < 0.7*n.distance:
				good.append((m.trainIdx, m.queryIdx))
				goodDMatch.append(m)

		if len(good) > 4:
			print(len(good))
			pointsCurrent = imageSet2['kp'] #query
			pointsPrevious = imageSet1['kp'] #train

			#KeyPoint.pt - point2D of the KeyPoint
			matchedPointsCurrent = np.float32(
				[pointsCurrent[i].pt for (__, i) in good]
			)
			matchedPointsPrev = np.float32(
				[pointsPrevious[i].pt for (i, __) in good]
				)

			#returns the 3x3 perspective transformation H on the source[xi,yi,1] to reach the destination planes[xf,yf,1]
			H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
			print(matchedPointsCurrent.shape, matchedPointsPrev.shape ,s.shape, np.count_nonzero(s == 1))


			if(pltflag==1):
				matchesMask = s.ravel().tolist()
				self.plotMatches(i2,imageSet2['kp'],i1,imageSet1['kp'],goodDMatch, matchesMask)
			return H
		return None

	def getSURFFeatures(self, im):
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp, des = self.surf.detectAndCompute(gray, None)
		return {'kp':kp, 'des':des}

	def featuresInMask(self, i1, intersection)	:
		imageSet = self.getSURFFeatures(i1)
		intersection = intersection/np.max(intersection)
		nf=0
		# fx=0
		# fy=0
		# for each in imageSet['kp']:
		# 	x=each.pt[0]
		# 	y=each.pt[1]
		# 	if(x>i1.shape[1]):
		# 		print(x-i1.shape[0])
		# 		fx=fx+1
		# 	if(y>i1.shape[0]):
		# 		fy=fy+1
		# print(fx,fy, i1.shape)

		# print(intersection.shape, i1.shape)
 
		# print(len(imageSet['kp']))

		for each in imageSet['kp']:
			# print(each.pt)
			y=int(each.pt[0])
			x=int(each.pt[1])
			if(intersection[x,y]==1):
				nf=nf+1
		return nf


	def plotMatches(self, img1,kp1,img2,kp2,good, matchesMask):
		draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   singlePointColor = None,
				   matchesMask = matchesMask, # draw only inliers
				   flags = 2)

		img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

		# plt.imshow(img3, 'gray'),plt.show()

		draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   singlePointColor = None,
				   flags = 2)

		img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

		# plt.imshow(img3, 'gray'),plt.show()