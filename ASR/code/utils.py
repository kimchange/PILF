#from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from skimage import metrics

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        file_list = os.listdir(dataset_dir)
        item_num = len(file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1
        file_name = [dataset_dir + '/%06d' % index + '.h5']
        with h5py.File(file_name[0], 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            data, label = augmentation(data, label)
            data, label = np.transpose(data, (1, 0)), np.transpose(label, (1, 0))
            data = ToTensor()(data.copy())
            label = ToTensor()(label.copy())
        return data, label

    def __len__(self):
        return self.item_num


def MultiTestSetDataLoader(args):
    dataset_dir = args.testset_dir
    data_list = os.listdir(dataset_dir)
    test_Loaders = []
    length_of_tests = 0

    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name)
        length_of_tests += len(test_Dataset)
        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=0, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name = 'ALL'):
        super(TestSetDataLoader, self).__init__()
        self.angin = args.angin
        self.dataset_dir = args.testset_dir + data_name
        self.file_list = []
        tmp_list = os.listdir(self.dataset_dir)
        for index, _ in enumerate(tmp_list):
            tmp_list[index] = tmp_list[index]
        self.file_list.extend(tmp_list)
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = self.dataset_dir + '/' + self.file_list[index]
        with h5py.File(file_name, 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            data, label = np.transpose(data, (1, 0)), np.transpose(label, (1, 0))
            data, label = ToTensor()(data.copy()), ToTensor()(label.copy())

        return data, label

    def __len__(self):
        return self.item_num


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5: # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    return data, label


def LFdivide(data, angRes, patch_size, stride):
    uh, vw = data.shape
    h0 = uh // angRes
    w0 = vw // angRes
    bdr = (patch_size - stride) // 2
    h = h0 + 2 * bdr
    w = w0 + 2 * bdr
    if (h - patch_size) % stride:
        numU = (h - patch_size)//stride + 2
    else:
        numU = (h - patch_size)//stride + 1
    if (w - patch_size) % stride:
        numV = (w - patch_size)//stride + 2
    else:
        numV = (w - patch_size)//stride + 1
    hE = stride * (numU-1) + patch_size
    wE = stride * (numV-1) + patch_size

    dataE = torch.zeros(hE*angRes, wE*angRes)
    for u in range(angRes):
        for v in range(angRes):
            Im = data[u*h0:(u+1)*h0, v*w0:(v+1)*w0]
            dataE[u*hE : u*hE+h, v*wE : v*wE+w] = ImageExtend(Im, bdr)
    subLF = torch.zeros(numU, numV, patch_size*angRes, patch_size*angRes)
    for kh in range(numU):
        for kw in range(numV):
            for u in range(angRes):
                for v in range(angRes):
                    uu = u*hE + kh*stride
                    vv = v*wE + kw*stride
                    subLF[kh, kw, u*patch_size:(u+1)*patch_size, v*patch_size:(v+1)*patch_size] = dataE[uu:uu+patch_size, vv:vv+patch_size]
    return subLF


def ImageExtend(Im, bdr):
    h, w = Im.shape
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[h - bdr: 2 * h + bdr, w - bdr: 2 * w + bdr]

    return Im_out


def LFintegrate(subLF, angRes, pz, stride, h0, w0):
    numU, numV, pH, pW = subLF.shape
    ph, pw = pH //angRes, pW //angRes
    bdr = (pz - stride) //2
    temp = torch.zeros(stride*numU, stride*numV)
    outLF = torch.zeros(angRes, angRes, h0, w0)
    for u in range(angRes):
        for v in range(angRes):
            for ku in range(numU):
                for kv in range(numV):
                    temp[ku*stride:(ku+1)*stride, kv*stride:(kv+1)*stride] = subLF[ku, kv, u*ph+bdr:u*ph+bdr+stride, v*pw+bdr:v*ph+bdr+stride]
                    '''
                    if ku != 0 and ku != numU-1 and kv !=0 and kv != numV-1 :
                        ##前一半
                        overlap_f_1 = subLF[ku, kv, u*ph+bdr+stride//2:u*ph+bdr+stride, v*pw+bdr:v*ph+bdr+stride]
                        #print(overlap_b_1.shape)
                        overlap_f_2 = subLF[ku+1, kv, u*ph:u*ph+bdr, v*pw+bdr:v*ph+bdr+stride]
                        overlap_f = (overlap_f_1 + overlap_f_2)/2
                        temp[ku*stride+stride//2:(ku+1)*stride, kv*stride:(kv+1)*stride] =  overlap_f
                        
                        ##后一半
                        overlap_b_1 = subLF[ku-1, kv, u*ph+bdr+stride:u*ph+bdr+stride+stride//2, v*pw+bdr:v*ph+bdr+stride]
                        #print(overlap_b_1.shape)
                        overlap_b_2 = subLF[ku, kv, u*ph+bdr:u*ph+bdr+stride//2, v*pw+bdr:v*ph+bdr+stride]
                        #print(overlap_b_2.shape)
                        overlap_b = (overlap_b_1 + overlap_b_2)/2
                        temp[ku*stride:(ku)*stride+stride//2, kv*stride:(kv+1)*stride] =  overlap_b
                        
                        ##上一半
                        overlap_t_1 = subLF[ku, kv, u*ph+bdr:u*ph+bdr+stride, v*ph+bdr+stride//2:v*ph+bdr+stride]
                        #print(overlap_b_1.shape)
                        overlap_t_2 = subLF[ku, kv+1, u*ph+bdr:u*ph+bdr+stride, v*pw:v*ph+bdr]
                        #print(overlap_b_2.shape)
                        overlap_t = (overlap_t_1 + overlap_t_2)/2
                        temp[ku*stride:(ku+1)*stride, kv*stride+stride//2:(kv+1)*stride] =  overlap_t
                        
                        ##下一半
                        overlap_bt_1 = subLF[ku, kv-1, u*ph+bdr:u*ph+bdr+stride, v*ph+bdr+stride:v*ph+bdr+stride+stride//2]
                        #print(overlap_b_1.shape)
                        overlap_bt_2 = subLF[ku, kv, u*ph+bdr:u*ph+bdr+stride, v*pw+bdr:v*ph+bdr+stride//2]
                        #print(overlap_b_2.shape)
                        overlap_bt = (overlap_bt_1 + overlap_bt_2)/2
                        temp[ku*stride:(ku+1)*stride, kv*stride:(kv)*stride+stride//2] =  overlap_bt 
                    '''
            outLF[u, v, :, :] = temp[0:h0, 0:w0]

    return outLF


def cal_psnr(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)

def cal_ssim(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True, data_range=1.0)

def cal_metrics(img1, img2, angRes, indicate):
    if len(img1.size())==2:
        [H, W] = img1.size()
        img1 = img1.view(angRes, H // angRes, angRes, W // angRes).permute(0,2,1,3)
    if len(img2.size())==2:
        [H, W] = img2.size()
        img2 = img2.view(angRes, H // angRes, angRes, W // angRes).permute(0,2,1,3)

    [U, V, h, w] = img1.size()
    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')

    bd = 22

    for u in range(U):
        for v in range(V):
            k = u*U + v
            if k in indicate:
                PSNR[u, v] = 0
            else:                
                PSNR[u, v] = cal_psnr(img1[u, v, bd:-bd, bd:-bd], img2[u, v, bd:-bd, bd:-bd])
                SSIM[u, v] = cal_ssim(img1[u, v, bd:-bd, bd:-bd], img2[u, v, bd:-bd, bd:-bd])
            pass
        pass

    psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    ssim_mean = SSIM.sum() / np.sum(SSIM > 0)

    return psnr_mean, ssim_mean
def cal_metrics_EPI(img1, img2, angRes_out, indicate):
    if len(img1.size())==2:
        [H, W] = img1.size()
        img1 = img1.view(angRes_out, H // angRes_out, angRes_out, W // angRes_out).permute(0,2,1,3)
    if len(img2.size())==2:
        [H, W] = img2.size()
        img2 = img2.view(angRes_out, H // angRes_out, angRes_out, W // angRes_out).permute(0,2,1,3)

    [U, V, H, W] = img1.size()

    bd = 22
    imag1 = img1[:, :, bd:-bd, bd:-bd]
    imag2 = img2[:, :, bd:-bd, bd:-bd]
    #epi_h = x_mv.view(b,self.angRes,self.angRes,h,w).permute(0,1,3,2,4).contiguous().view(-1,1,self.angRes,w)
    #epi_w = epi_h_hr.contiguous().view(b, self.angRes, h, self.angRes_out, w).permute(0,3,4,1,2)
    
    imag1_EPI_h = imag1.permute(0,2,1,3).contiguous().view(-1,V,W-2*bd)
    imag2_EPI_h = imag2.permute(0,2,1,3).contiguous().view(-1,V,W-2*bd)
    PSNR = np.zeros(shape=(U, (H-2*bd)), dtype='float32')
    SSIM = np.zeros(shape=(U, (H-2*bd)), dtype='float32')

    '''
    imag1_EPI_v = imag1.permute(1,3,0,2).contiguous().view(-1,U,H-2*bd)
    imag2_EPI_v = imag2.permute(1,3,0,2).contiguous().view(-1,U,H-2*bd)
    PSNR = np.zeros(shape=(V, (W-2*bd)), dtype='float32')
    SSIM = np.zeros(shape=(V, (W-2*bd)), dtype='float32')
    '''

    for u in range(U):
        for h in range(H-2*bd):
            k = u*U + h
            PSNR[u, h] = cal_psnr(imag1_EPI_h[k,:,:], imag2_EPI_h[k,:,:])
            SSIM[u, h] = cal_ssim(imag1_EPI_h[k,:,:], imag2_EPI_h[k,:,:])
            # if (u == 0 or u == 7):# and (h == 0 or h == H-2*bd-1)
            #     continue
            #     #print(imag1_EPI_h[k,1:-1,:].shape)
            #     #PSNR[u, h] = cal_psnr(imag1_EPI_h[k,1:-1,:], imag2_EPI_h[k,1:-1,:])
            #     #SSIM[u, h] = cal_ssim(imag1_EPI_h[k,1:-1,:], imag2_EPI_h[k,1:-1,:])
            # else:
            #     print(imag1_EPI_h[k,:,:].shape)
            #     PSNR[u, h] = cal_psnr(imag1_EPI_h[k,:,:], imag2_EPI_h[k,:,:])
            #     SSIM[u, h] = cal_ssim(imag1_EPI_h[k,:,:], imag2_EPI_h[k,:,:])
            # pass
        pass
    '''
    for v in range(V):
        for w in range(W-2*bd):
            k = v*V + w
            PSNR[v, w] = cal_psnr(imag1_EPI_v[k,:,:], imag2_EPI_v[k,:,:])
            SSIM[v, w] = cal_ssim(imag1_EPI_v[k,:,:], imag2_EPI_v[k,:,:])
            pass
        pass
    '''
    psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    ssim_mean = SSIM.sum() / np.sum(SSIM > 0)

    return psnr_mean, ssim_mean
def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st
