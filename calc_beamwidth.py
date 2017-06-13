import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import h5py

def cor_circle(data,r):
	#フィルター作成
	#中心からの距離を成分とする行列作成
	y,x = np.ogrid[-r:r+1, -r:r+1]
	mask = x*x + y*y <= r*r

	farray = np.zeros((2*r+1, 2*r+1))
	farray[mask] = 1

	ave_data = cv2.filter2D(data,-1,farray)
	#ave_data2 = cv2.cvtColor(ave_data, cv2.COLOR_BGR2RGB)
	return ave_data

def cor_height(data,r):
	#フィルター作成
	#横方向rマスの移動平均
	farray = np.ones(r)
	ave_data = cv2.filter2D(data,-1,farray)
	#ave_data2 = cv2.cvtColor(ave_data, cv2.COLOR_BGR2RGB)
	return ave_data

def cor_width(data,r):
	#フィルター作成
	#縦方向rマスの移動平均
	farray = np.ones((1,r))
	
	ave_data = cv2.filter2D(data,-1,farray)

	return ave_data

if __name__ == "__main__":
	input_file = "11.bgdata"
	h5file = h5py.File(input_file,"r")
	folder = "BG_DATA/1/"
	data = np.array(h5file[folder+"DATA"].value)/100000
	data.shape = (1200,1600)
	#data = np.ones((1200,1600))
	r = 300
	ave_data = cor_circle(data,r)
	height = cor_width(data,r)
	plt.imshow(ave_data, interpolation='nearest')
	plt.show()