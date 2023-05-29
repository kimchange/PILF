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
        # file_list = file_list[0:12] # to test if the code can run
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
            data = ToTensor()(data.copy())
            label = ToTensor()(label.copy())
        return data, label

    def __len__(self):
        return self.item_num
    

class TrainSetLoader_cache_inmemory(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader_cache_inmemory, self).__init__()
        self.dataset_dir = dataset_dir
        file_list = os.listdir(dataset_dir)
        # file_list = file_list[0:12] # to test if the code can run
        item_num = len(file_list)
        self.item_num = item_num

        self.data =[]
        self.label = []

        for ii in range(item_num):
            file_name = [dataset_dir + '/%06d' % (ii+1) + '.h5']
            with h5py.File(file_name[0], 'r') as hf:
                self.data.append(np.array(hf.get('data'))) 
                self.label.append(np.array(hf.get('label'))) 

        # file_name = [dataset_dir + '/%06d' % index + '.h5']

    def __getitem__(self, index):
        data = self.data[index % self.item_num]
        label = self.label[index % self.item_num]

        # dataset_dir = self.dataset_dir
        # index = index + 1
        # file_name = [dataset_dir + '/%06d' % index + '.h5']
        # with h5py.File(file_name[0], 'r') as hf:
        #     data = np.array(hf.get('data'))
        #     label = np.array(hf.get('label'))
        data, label = augmentation(data, label)
        data = ToTensor()(data.copy())
        label = ToTensor()(label.copy())
        return data, label

    def __len__(self):
        return self.item_num

class TrainSetLoader_k(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader_k, self).__init__()
        self.dataset_dir = dataset_dir
        # file_list = os.listdir(dataset_dir)
        self.file_list = []
        self.inp_size = 32
        self.scale = 4
        self.data_list = ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'INRIA_Lytro', 'Stanford_Gantry']

        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        item_num = len(self.file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1
        file_name = [self.dataset_dir + self.file_list[index % self.item_num]]
        with h5py.File(file_name[0], 'r') as hf:
            LF = np.array(hf.get('LF'))  # LF image
        
        h,w = self.inp_size, self.inp_size
        H,W = round(h*self.scale), round(w*self.scale)
        H0 = random.randint(0, LF.shape[-2] - H)
        W0 = random.randint(0, LF.shape[-1] - W)
        LF = LF[:,:,:,H0:H0+H, W0:W0+W]
        LF = torch.tensor(LF)

        LF = self.augmentation(LF)

        C,U,V,H,W = LF.shape

        label = LF_rgb2ycbcr(LF.unsqueeze(0))[0,0:1,:,:,:,:]

        LF = LF.reshape(C,U*V,H,W).permute(1,0,2,3)

        LFLR = Bicubic()(LF, 1 / self.scale).permute(1,0,2,3).reshape(C,U,V,H//4,W//4)

        data = LF_rgb2ycbcr(LFLR.unsqueeze(0))[0,0:1,:,:,:,:] # 1,u,v,h,w

        data = data.permute(0,1,3,2,4).reshape(1, U*H//4, V*W//4)
        label = label.permute(0,1,3,2,4).reshape(1, U*H, V*W)


        # with h5py.File(file_name[0], 'r') as hf:
        #     data = np.array(hf.get('data'))
        #     label = np.array(hf.get('label'))
        #     data, label = augmentation(data, label)
        #     data = ToTensor()(data.copy())
        #     label = ToTensor()(label.copy())

        return data, label

    @staticmethod
    def augmentation(label):
        if random.random() < 0.5:  # flip along W-V direction
            label = torch.flip(label, dims=[2, 4])
        if random.random() < 0.5:  # flip along W-V direction
            label = torch.flip(label, dims=[1, 3])
        if random.random() < 0.5:  # transpose between U-V and H-W
            label = label.permute(0, 2, 1, 4, 3)
        return label

    def __len__(self):
        return self.item_num



def MultiTestSetDataLoader(args):
    dataset_dir = args.testset_dir
    data_list = sorted(os.listdir(dataset_dir))
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
        self.angRes = args.angRes
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

            outLF[u, v, :, :] = temp[0:h0, 0:w0]

    return outLF


def cal_psnr(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.peak_signal_noise_ratio(img1_np, img2_np)#, data_range=1.0

def cal_ssim(img1, img2):
    img1_np = img1.data.cpu().numpy().clip(0, 1)
    img2_np = img2.data.cpu().numpy().clip(0, 1)

    return metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True)#, data_range=1.0)

def cal_ssim_1(img1, img2):
    img1_np = img1.data.cpu().numpy().clip(0, 1)
    img2_np = img2.data.cpu().numpy().clip(0, 1)

    return metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True, data_range=1.0)

def cal_metrics(img1, img2, angRes):
    if len(img1.size())==2:
        [H, W] = img1.size()
        img1 = img1.view(angRes, H // angRes, angRes, W // angRes).permute(0,2,1,3)
    if len(img2.size())==2:
        [H, W] = img2.size()
        img2 = img2.view(angRes, H // angRes, angRes, W // angRes).permute(0,2,1,3)

    [U, V, h, w] = img1.size()
    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')
    SSIM_1 = np.zeros(shape=(U, V), dtype='float32')

    if angRes==5:
        for u in range(U):
            for v in range(V):
                PSNR[u, v] = cal_psnr(img1[u, v, :, :], img2[u, v, :, :])
                SSIM[u, v] = cal_ssim(img1[u, v, :, :], img2[u, v, :, :])
                SSIM_1[u, v] = cal_ssim_1(img1[u, v, :, :], img2[u, v, :, :])
                pass
            pass
    else:
        bd = 22
        for u in range(U):
            for v in range(V):
                PSNR[u, v] = cal_psnr(img1[u, v, bd:-bd, bd:-bd], img2[u, v, bd:-bd, bd:-bd])
                SSIM[u, v] = cal_ssim(img1[u, v, bd:-bd, bd:-bd], img2[u, v, bd:-bd, bd:-bd])
                SSIM_1[u, v] = cal_ssim_1(img1[u, v, bd:-bd, bd:-bd], img2[u, v, bd:-bd, bd:-bd])
                pass
            pass


    psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    ssim_mean = SSIM.sum() / np.sum(SSIM > 0)
    ssim_mean_1 = SSIM_1.sum() / np.sum(SSIM > 0)

    return psnr_mean, ssim_mean#, ssim_mean_1




# implementation of matlab bicubic interpolation in pytorch
class Bicubic(object):
    def __init__(self):
        super(Bicubic, self).__init__()

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32)
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32)

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        P = np.ceil(kernel_width) + 2

        indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)
        indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        indice0 = torch.min(torch.max(torch.FloatTensor([1]), indice0), torch.FloatTensor([in_size[0]])).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]), indice1), torch.FloatTensor([in_size[1]])).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def __call__(self, input, scale=1/4):
        b, c, h, w = input.shape

        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)
        weight0 = weight0[0].to(input.device)
        weight1 = weight1[0].to(input.device)

        indice0 = indice0[0].long()
        indice1 = indice1[0].long()

        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = torch.sum(out, dim=3)
        A = out.permute(0, 1, 3, 2)

        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = out.sum(3).permute(0, 1, 3, 2)

        return out



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