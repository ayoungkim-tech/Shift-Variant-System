#!/home/jorgehdsp/.venv/tensorflow/bin/python
import sys

sys.path.append('/home/jorgehdsp/.venv/tensorflow/bin/python')
import tensorflow as tf
import os
from matplotlib import pyplot as plt

from os import listdir
from os.path import isfile, join
from random import choices
import numpy as np


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#-------------------- This are main py------------------------
from Models_and_Tools.Read_Spectral import *   # data set build
from Models_and_Tools.recoverynet_super import *     # Net build

def root_mean_squared_error(y_true, y_pred):
    return 20*tf.reduce_mean(tf.norm(y_true - y_pred, ord=2, axis=-1)) + tf.reduce_mean(tf.norm(y_true - y_pred, ord='fro', axis=[1,2]))

def PSNR_Metric(y_true, y_pred):
  return tf.reduce_mean(tf.image.psnr(y_true,y_pred,1))

#----------------------------- directory of the spectral data set -------------------------
PATH = '../dataset/Formated/'
PATH2 = '../dataset/Formated/'
# parameters of the net
#BATCH_SIZE = 4; IMG_WIDTH = 250; IMG_HEIGHT = 250; L_bands    = 25; L_imput    = 25
BATCH_SIZE = 4; IMG_WIDTH = 512; IMG_HEIGHT = 482; L_bands    = 25; L_imput    = 25

#---------------------- Important Parameters to Compare --------------------------------------
diam = 3e-6
name_file = 'pretrained_networks/modelo_clip_3mm_5cm.h5'
clip = 0
wrap = 1 # see in the propagation change
File_name = 'R_len3_v2'
try:
    os.stat(File_name)
except:
    os.mkdir(File_name)

model = Proposed_net(input_size=(IMG_HEIGHT,IMG_WIDTH,L_imput),depth=L_imput,diam=diam)      # build the Net from recoverynet.py
model.load_weights(name_file)
model.compile(run_eagerly=True)
model_psfs  = Psf_show(input_size=(IMG_HEIGHT,IMG_WIDTH,L_imput),depth=L_imput,diam=diam)
it=1
model_psfs.layers[it].set_weights(model.layers[it].get_weights())

# See the height_map
temporal = model_psfs.get_weights()
scipy.io.savemat(File_name + "/colfilt.mat", {'colfilt': temporal[1]})
polinomios=temporal[0]
zernike_volume = np.load('zernike_volume1_%d.npy' % 1000)
height_map = np.sum(polinomios * zernike_volume, axis=0)
scipy.io.savemat(File_name + "/Height_before.mat",{'height_map': height_map})
plt.figure()
plt.imshow(height_map),plt.title('Height Map_before')
plt.colorbar()
plt.savefig(File_name+'/Height_before.png')
plt.show()


if(clip==1):
    height_map = tf.clip_by_value(height_map, -0.755e-6, 0.755e-6).numpy()
if(wrap==1):
    #height_map = ((((height_map / 0.755e-6) * np.pi + np.pi) % (2 * np.pi) - np.pi) / np.pi) * 0.755e-6
    height_map = tf.math.floormod(height_map + 0.755e-6, 2*0.755e-6 ) - 0.755e-6

height_map = height_map-np.min(height_map)
scipy.io.savemat(File_name + "/Height.mat",{'height_map': height_map.numpy()})
plt.figure()
plt.imshow(height_map),plt.title('Height Map')
plt.colorbar()
plt.savefig(File_name+'/Height.png')
plt.show()



## See some reconstruction
Img_spectral = loadmat('dataset/NTIRE2020_Validation_Spectral/ARAD_HS_0464.mat')
#Img_spectral = loadmat('dataset/ARAD_HS_0464.mat')
Ref_img = Img_spectral['cube'][:,:,3:-3]#[0:250,0:250,3:-3]
Ref_img = Ref_img/np.max( Ref_img)
plt.figure()
temp = Ref_img[:,:,[20,10,3]]/np.max( Ref_img[:,:,[20,10,3]])
scipy.io.savemat(File_name + "/gt.mat",{'Ref_img': Ref_img})
plt.subplot(1,3,2),plt.imshow(temp),plt.title('ground truth')
Ref_img=np.expand_dims(Ref_img,0)
Temp = model_psfs.predict(Ref_img,batch_size=1)
Temp[0][0,:,:,:] = Temp[0][0,:,:,:]/np.max(Temp[0][0,:,:,:])
scipy.io.savemat(File_name + "/measurement.mat",{'measurement': Temp[0][0,:,:,:]})
plt.subplot(1,3,1),plt.imshow(Temp[0][0,:,:,:]),plt.title('Measurement')
Recon= model.predict(Ref_img,batch_size=1)
Recon=Recon[0,:,:,:]
scipy.io.savemat(File_name + "/Recovered.mat",{'Recovered': Recon})
temp1 = Recon[:,:,[20,10,3]]/np.max(np.abs(Recon[:,:,[20,10,3]]))
plt.subplot(1,3,3),plt.imshow(temp1),plt\
    .title('Recovered ='+str(PSNR_Metric(Ref_img,Recon).numpy()))
plt.savefig(File_name+'/result.png')
plt.show()

## See some reconstruction
# # Img_spectral = loadmat('dataset/Butterfly.mat')
# # Ref_img = Img_spectral['Iin'][150:150+250,150:150+250,0:-6]
# Img_spectral = loadmat('dataset/Butterfly_250.mat')
# Ref_img = Img_spectral['img'][:,:,:-6]
# Ref_img = Ref_img/np.max(Ref_img)
# plt.figure()
# temp = Ref_img[:,:,[20,10,3]]/np.max( Ref_img[:,:,[20,10,3]])
# scipy.io.savemat(File_name + "/gt_b.mat",{'Ref_img': Ref_img})
# plt.subplot(1,3,2),plt.imshow(temp),plt.title('ground truth butterfly')
# Ref_img=np.expand_dims(Ref_img,0)
# Temp = model_psfs.predict(Ref_img,batch_size=1)
# Temp[0][0,:,:,:] = Temp[0][0,:,:,:]/np.max(Temp[0][0,:,:,:])
# scipy.io.savemat(File_name + "/measurement_b.mat",{'measurement': Temp[0][0,:,:,:]})
# plt.subplot(1,3,1),plt.imshow(Temp[0][0,:,:,:]),plt.title('Measurement butterfly')
# Recon= model.predict(Ref_img,batch_size=1)
# Recon=Recon[0,:,:,:]
# scipy.io.savemat(File_name + "/Recovered_b.mat",{'Recovered': Recon})
# temp1 = Recon[:,:,[20,10,3]]/np.max(np.abs(Recon[:,:,[20,10,3]]))
# plt.subplot(1,3,3),plt.imshow(temp1),plt\
#     .title('Recovered ='+str(PSNR_Metric(Ref_img,Recon).numpy()))
# plt.savefig(File_name+'/result_b.png')
# plt.show()