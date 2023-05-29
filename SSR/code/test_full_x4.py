import time
import argparse
import scipy.misc
import torch.backends.cudnn as cudnn
import os
import sys

from utils import *

#from model_HLFSR_re import Net
# from LFT import Net
from tqdm import tqdm
import scipy.io as sio
import time
from einops import rearrange
import torch
import torch.nn.functional as F
import h5py
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upscale_factor", type=int, default=4, help="upscale factor")
    parser.add_argument("--channels", type=int, default=64, help="number of channels")
    parser.add_argument('--model_name', type=str, default='C42net')

    parser.add_argument('--testset_dir', type=str, default='../data_for_test/TestData_4xSR_5x5/')


    parser.add_argument("--crop_test_method",type=int, default=4, help="cropped test method( 1- whole image| 2- cropped mxn patches | 3- cropped 4 patches")
    parser.add_argument("--patchsize", type=int, default=64, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--stride", type=int, default=32, help="The stride between two test patches is set to patchsize/2")

    # parser.add_argument('--codefolder', type=str, default='/home/slfm/kimchange/LFSR/save/checkpoint_20230314-6C42conv-6C42trans-c64/code/')
    parser.add_argument('--codefolder', type=str, default='../save/checkpoint_20230327/code/')


    # parser.add_argument('--model_path', type=str, default='/home/slfm/kimchange/LFSR/save/checkpoint_20230309-4-convs-64-4-trans-6-noposition/C42net_4xSR_5x5_epoch_50.pth.tar')
    # parser.add_argument('--model_path', type=str, default="/home/slfm/kimchange/LFSR/save/checkpoint_20230314-6C42conv-6C42trans-c64/C42net_4xSR_5x5_epoch_38.pth.tar")
    parser.add_argument('--model_path', type=str, default="../save/checkpoint_20230327/C42net_4xSR_5x5_epoch_63.pth.tar")
    # parser.add_argument('--save_path', type=str, default='./ValResults/')
    parser.add_argument('--save_path', type=str, default='../TestResultsx4-full-pad-ep63/')
    parser.add_argument('--tag', type=str, default='')

    args = parser.parse_args()

    sys.path.append(args.codefolder)
    
    

    return args


cfg = parse_args()
from c42conv_epixt import Net



def test(cfg, test_Names, test_loaders):

    #net = EDSR(cfg.angRes, cfg.upscale_factor)
    net = Net(cfg.angRes, cfg.upscale_factor)
    #net = net2x()
    net.to(cfg.device)
    cudnn.benchmark = True

    if os.path.isfile(cfg.model_path):
        # model = torch.load(cfg.model_path, map_location={'cuda:1': cfg.device})
        # net.load_state_dict(model['state_dict'])
        net.load_state_dict(torch.load('/home/slfm/as13000/kimchange/LFSR/save/checkpoint_20230327/C42net_4xSR_5x5_epoch_63_state_dict.pth.tar'))
    else:
        print("=> no model found at '{}'".format(cfg.model_path))
        pass

    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        num_testset = []
        for index, test_name in enumerate(test_Names):

            flag = 0
            test_loader = test_loaders[index]
            # outLF = inference(test_loader, test_name, net, flag)
            psnr_epoch_test, ssim_epoch_test, num_epoch_test = inference(test_loader,test_name, net, flag)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            num_testset.append(num_epoch_test)
            # print('Finishing testing----%15s' % (test_name))

            print(time.ctime()[4:-5] + ' Test----%15s,\t test Number---%d, PSNR---%f, SSIM---%f' % (test_name, num_epoch_test, psnr_epoch_test, ssim_epoch_test))
            txtfile = open(cfg.save_path + cfg.tag + cfg.model_name + '_test.txt', 'a')
            txtfile.write('Dataset----%10s,\t test Number---%d ,\t PSNR---%f,\t SSIM---%f\n' % (test_name, num_epoch_test, psnr_epoch_test, ssim_epoch_test))
            txtfile.close()


            pass

        psnr_avg = sum([psnr_testset[ii]*num_testset[ii] for ii in range(len(num_testset))]) / sum(num_testset)
        ssim_avg = sum([ssim_testset[ii]*num_testset[ii] for ii in range(len(num_testset))]) / sum(num_testset)

        txtfile = open(cfg.save_path + cfg.tag + cfg.model_name + '_test.txt', 'a')
        txtfile.write('Total testset,\t test Number---%d ,\t PSNR---%f,\t SSIM---%f\n' % (sum(num_testset), psnr_avg, ssim_avg))
        txtfile.close()
        pass


def inference(test_loader, test_name, net, flag):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        b, c, h, w = data.size()
        
        if cfg.crop_test_method == 4:
            print('Testing with full resolution')
            data = data.to(cfg.device)

            label = label.to(cfg.device)
            data = rearrange(data, 'b c (u h) (v w) -> b (u v) c h w', u=cfg.angRes, v=cfg.angRes)
            b, n, c, h, w = data.size()


            bdr = 0 # pad length 0 for x4

            data = ImageExtend_batch2(data.reshape(b, n*c, h, w), bdr).reshape(b,n,c,h+bdr*2, w+bdr*2)
            
            
            # data = data.contiguous().view(b, n, c, h_e, w_e)
            scale  = cfg.upscale_factor


            data = rearrange(data, 'b (u v) c h w -> b c (u h) (v w)', u=cfg.angRes, v=cfg.angRes)
            out = net(data)



            out = rearrange(out, 'b c (u h) (v w) -> b (u v) c h w', u=cfg.angRes, v=cfg.angRes)

            outLF = rearrange(out, 'b (u v) c h w -> b u v c h w', u=cfg.angRes, v=cfg.angRes)
            outLF = outLF[:,:,:,:,bdr*scale:outLF.shape[-2]-bdr*scale,bdr*scale:outLF.shape[-1]-bdr*scale]

            outLF = rearrange(outLF, 'b u v c h w -> b c (u h) (v w)', u=cfg.angRes, v=cfg.angRes)

            psnr, ssim = cal_metrics(label, outLF, cfg.angRes)

            psnr_iter_test.append(psnr)
            ssim_iter_test.append(ssim)

            outLF = rearrange(outLF, 'b c (u h) (v w) -> b u v c h w', u=cfg.angRes, v=cfg.angRes)


            outLF = outLF.squeeze()

        isExists = os.path.exists(cfg.save_path + test_name)
        if not (isExists ):
            os.makedirs(cfg.save_path + test_name)

        # sio.savemat(cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
        #                 {'LF': outLF.cpu().numpy()})
        with h5py.File(cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.h5', 'w') as hf:
                    hf.create_dataset('LF', data=outLF.cpu().numpy(), dtype='single')
                    hf.close()
                    pass

        pass
    
    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())
    num_epoch_test = len(psnr_iter_test)
    return psnr_epoch_test, ssim_epoch_test, num_epoch_test

def ImageExtend_right(Im, bdr_h, bdr_w):
    b, c, h, w = Im.shape
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_down), dim=-2)

    Im_out = Im_Ext[:, :, : bdr_h, : bdr_w]
    #Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    #Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    #Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    #Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    #Im_out = Im_Ext[h - bdr: 2 * h + bdr, w - bdr: 2 * w + bdr]

    return Im_out


def ImageExtend_batch2(Im, bdr):
    b, c, h, w = Im.shape
    Im_out = F.pad(Im, (1, )*4, mode='replicate')
    Im_out = F.pad(Im_out, (bdr, )*4, mode='reflect')
    Im_out = Im_out[:,:,torch.arange(Im_out.shape[-2])!= bdr ]
    Im_out = Im_out[:,:,torch.arange(Im_out.shape[-2])!= bdr+h ]
    Im_out = Im_out[:,:,:,torch.arange(Im_out.shape[-1])!= bdr ]
    Im_out = Im_out[:,:,:,torch.arange(Im_out.shape[-1])!= bdr+w ]
    return Im_out




def main(cfg):
    # test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader_N(cfg)
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    # test_Names = test_Names[4:]
    # test_Loaders = test_Loaders[4:]
    # length_of_tests = length_of_tests-4
    test(cfg, test_Names, test_Loaders)


if __name__ == '__main__':
    # cfg = parse_args()
    
    # print(123)
    main(cfg)
