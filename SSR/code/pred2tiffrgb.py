import torch
import h5py
import scipy.io as scio
from tifffile import imwrite
import numpy as np
import os
import utils
# datasets = ['NTIRE_Val_Real', 'NTIRE_Val_Synth']
folder = '../TestResultsx4-full-pad-ep63'
# folder = '../TestResultsx2-full-pad8ifLytro-ep60'
datasets = ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry'] #os.listdir(folder)
savefolder = folder + '_tiffrgb'
labelfolder = '../LFSSR_data/datasets'
angRes = 5

savevolume = 0
calc_psnrssim = 1

def LF_rgb2ycbcr(x):
    y = torch.zeros_like(x)
    y[:,0,:,:,:,:] =  65.481 * x[:,0,:,:,:,:] + 128.553 * x[:,1,:,:,:,:] +  24.966 * x[:,2,:,:,:,:] +  16.0
    y[:,1,:,:,:,:] = -37.797 * x[:,0,:,:,:,:] -  74.203 * x[:,1,:,:,:,:] + 112.000 * x[:,2,:,:,:,:] + 128.0
    y[:,2,:,:,:,:] = 112.000 * x[:,0,:,:,:,:] -  93.786 * x[:,1,:,:,:,:] -  18.214 * x[:,2,:,:,:,:] + 128.0

    y = y / 255.0
    return y


def LF_ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = torch.zeros_like(x)
    y[:,0,:,:,:,:] = mat_inv[0,0] * x[:,0,:,:,:,:] + mat_inv[0,1] * x[:,1,:,:,:,:] + mat_inv[0,2] * x[:,2,:,:,:,:] - offset[0]
    y[:,1,:,:,:,:] = mat_inv[1,0] * x[:,0,:,:,:,:] + mat_inv[1,1] * x[:,1,:,:,:,:] + mat_inv[1,2] * x[:,2,:,:,:,:] - offset[1]
    y[:,2,:,:,:,:] = mat_inv[2,0] * x[:,0,:,:,:,:] + mat_inv[2,1] * x[:,1,:,:,:,:] + mat_inv[2,2] * x[:,2,:,:,:,:] - offset[2]
    return y


for dataset in datasets:
    if not os.path.exists(savefolder+'/'+dataset+'/'):
        try:
            os.mkdir(savefolder+'/'+dataset+'/') 
        except:
            os.makedirs(savefolder+'/'+dataset+'/') 

psnr_testset = []
ssim_testset = []
num_testset = []

for dataset in datasets:
    labels = os.listdir(labelfolder+'/'+dataset+'/test')
    psnr_file = []
    ssim_file = []

    if calc_psnrssim:
        txtfile = open(savefolder + '/y_psnrssim.txt', 'a')
        txtfile.write('Dataset----%10s :\n' % (dataset))
        txtfile.close()
        print('Dataset----%10s :\n' % (dataset))

    for label in labels:
        try:
            label_rgbh = h5py.File(labelfolder+'/'+dataset+'/test'+'/'+label, 'r')
            label_rgbn = np.array( label_rgbh.get('LF') ).transpose((4, 3, 2, 1, 0)) # u,v,h,w,c
        except:
            label_rgbh = scio.loadmat(labelfolder+'/'+dataset+'/test'+'/'+label)
            label_rgbn = np.array( label_rgbh.get('LF') ) # u,v,h,w,c
        (U, V, H, W, _) = label_rgbn.shape
        H = H // 4 * 4
        W = W // 4 * 4

        # Extract central angRes * angRes views
        label_rgbn = label_rgbn[(U-angRes)//2:(U+angRes)//2, (V-angRes)//2:(V+angRes)//2, 0:H, 0:W, 0:3]
        (U, V, H, W, _) = label_rgbn.shape
        label_rgbn = np.transpose(label_rgbn, (4,0,1,2,3)) # c,u,v,h,w
        # label_rgbh = h5py.File(labelfolder+'/'+dataset+'/test'+'/'+label, 'r')
        # label_rgbn = np.array( label_rgbh.get('LF') ) # c,w,h,v,u
        # label_rgbn = np.transpose(label_rgbn, (0,4,3,2,1)) # c,u,v,h,w
        label_rgbt = torch.tensor(label_rgbn).unsqueeze(0) # 1,c,u,v,h,w
        label_ycbcrt = LF_rgb2ycbcr(label_rgbt)


        pred_ycbcrt = label_ycbcrt
        pred_yh = h5py.File(folder+'/'+dataset+'/'+label[0:-4]+'.h5', 'r')
        pred_yn = np.array( pred_yh.get('LF') ) # u v h w
        pred_yt = torch.tensor(pred_yn) # u v h w
        if calc_psnrssim:
            psnr, ssim = utils.cal_metrics(label_ycbcrt[0,0,:,:,:,:], pred_yt, angRes)
            
            psnr_file.append(psnr)
            ssim_file.append(ssim)

            txtfile = open(savefolder + '/y_psnrssim.txt', 'a')
            txtfile.write('test file---%15s ,\t PSNR---%f,\t SSIM---%f\n' % (label, psnr, ssim))
            txtfile.close()
            print('test file---%15s ,\t PSNR---%f,\t SSIM---%f\n' % (label, psnr, ssim))


        if savevolume: 
            pred_ycbcrt[0,0,:,:,:,:] = pred_yt

            pred_rgbt = LF_ycbcr2rgb(pred_ycbcrt).squeeze(0) # c,u,v,h,w
            pred_rgbt = pred_rgbt.permute(1,2,0,3,4 ) # u,v,c,h,w

            pred_rgbn = np.array(pred_rgbt)
            pred_rgbn = np.reshape(pred_rgbn, [pred_rgbn.shape[0]*pred_rgbn.shape[1],pred_rgbn.shape[2], pred_rgbn.shape[3], pred_rgbn.shape[4]])

            imwrite(savefolder+'/'+dataset+'/'+label[0:-4]+'.tif', pred_rgbn, imagej=True, metadata={'axes': 'ZCYX'}, compression ='zlib')

    
    if calc_psnrssim:
        psnr_testset.append(float(np.array(psnr_file).mean()))
        ssim_testset.append(float(np.array(ssim_file).mean()))
        num_testset.append(len(psnr_file))
        txtfile = open(savefolder + '/y_psnrssim.txt', 'a')
        txtfile.write('Dataset----%10s,\t test Number---%d ,\t PSNR---%f,\t SSIM---%f\n' % (dataset, num_testset[-1], psnr_testset[-1], ssim_testset[-1]))
        txtfile.close()
        print('Dataset----%10s,\t test Number---%d ,\t PSNR---%f,\t SSIM---%f\n' % (dataset, num_testset[-1], psnr_testset[-1], ssim_testset[-1]))
        

if calc_psnrssim:
    psnr_avg = sum([psnr_testset[ii]*num_testset[ii] for ii in range(len(num_testset))]) / sum(num_testset)
    ssim_avg = sum([ssim_testset[ii]*num_testset[ii] for ii in range(len(num_testset))]) / sum(num_testset)
    txtfile = open(savefolder + '/y_psnrssim.txt', 'a')
    txtfile.write('Total testset,\t test Number---%d ,\t PSNR---%f,\t SSIM---%f\n' % (sum(num_testset), psnr_avg, ssim_avg))
    txtfile.close()
    print('Total testset,\t test Number---%d ,\t PSNR---%f,\t SSIM---%f\n' % (sum(num_testset), psnr_avg, ssim_avg))
        # lfh = h5py.File(folder+'/'+dataset+'/'+file, 'r')
        # lfn = np.array( lfh.get('LF') ) # c,w,h,v,u


        # lfn = np.transpose(lfn, (4,3,0,2,1)) # u,v,c,h,w
        # print(f'{dataset}/{file} lfn.shape is {lfn.shape}')
        # lfn = np.reshape(lfn, [lfn.shape[0]*lfn.shape[1],lfn.shape[2], lfn.shape[3], lfn.shape[4]])

        # imwrite(folder+'_tif/'+dataset+'/'+file[0:-4]+'.tif', lfn, imagej=True, metadata={'axes': 'ZCYX'}, compression ='zlib')

