import time
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils import *
# from LFT import Net
from c42conv_epixt import Net
# from LFT_4D import Net
# from LFT_epif import Net

import os
from tensorboardX import SummaryWriter


#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:6')
    parser.add_argument('--device_ids', type=str, default='6,7')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upscale_factor", type=int, default=2, help="upscale factor")
    parser.add_argument('--model_name', type=str, default='C42net')
    # parser.add_argument('--trainset_dir', type=str, default='/root/data1/LF_fix/Data/TrainingData_32_4_4xSR_5x5')
    # parser.add_argument('--trainset_dir', type=str, default='/home/slfm/kimchange/LFSR/data_for_training/SSR_5x5/')
    # parser.add_argument('--trainset_dir', type=str, default='/home/slfm/kimchange/LFSR/data_for_training/TrainingData_32_2_4xSR_5x5')
    # parser.add_argument('--trainset_dir', type=str, default='/mnt2/dataset/LFSSR_data/TrainingData_32_4_4xSR_5x5/')
    # # parser.add_argument('--testset_dir', type=str, default='/root/data1/LF_fix/Data/TestData_4xSR_5x5/')
    # # parser.add_argument('--testset_dir', type=str, default='/home/slfm/kimchange/LFSR/data_for_test/TestData_4xSR_5x5/')
    # parser.add_argument('--testset_dir', type=str, default='/mnt2/dataset/LFSSR_data/TestData_4xSR_5x5/')
    # parser.add_argument('--trainset_dir', type=str, default='/mnt2/dataset/LFSSR_data/TrainingData_2xSR_5x5/')
    # parser.add_argument('--testset_dir', type=str, default='/mnt2/dataset/LFSSR_data/TestData_2xSR_5x5/')
    # parser.add_argument('--trainset_dir', type=str, default='../data_for_training/TrainingData_32_4_4xSR_5x5/')
    # parser.add_argument('--testset_dir', type=str, default='../data_for_test/TestData_4xSR_5x5/')
    # parser.add_argument('--trainset_dir', type=str, default='/home/slfm/kimchange/LFSR/data_for_training/TrainingData_32_4_4xSR_5x5/')
    # parser.add_argument('--testset_dir', type=str, default='/home/slfm/kimchange/LFSR/data_for_test/TestData_4xSR_5x5/')
    parser.add_argument('--trainset_dir', type=str, default='/home/slfm/kimchange/LFSR/data_for_training/TrainingData_2xSR_5x5/')
    parser.add_argument('--testset_dir', type=str, default='/home/slfm/kimchange/LFSR/data_for_training/TestData_2xSR_5x5/')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')

    parser.add_argument("--patchsize", type=int, default=64, help="crop into patches for validation")
    parser.add_argument("--stride", type=int, default=32, help="stride for patch cropping")

    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--model_path', type=str, default='../save/checkpoint_20220326-c42conv-epixt-maxdisp2-layer6/C42net_2xSR_5x5_epoch_32.pth.tar')

    parser.add_argument('--tag', type=str, default='')
    

    return parser.parse_args()


def train(cfg, train_loader, test_Names, test_loaders):

    net = Net(cfg.angRes, cfg.upscale_factor)
    net.to(cfg.device)
    cudnn.benchmark = True
    epoch_state = 0
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'])
            optimizer.load_state_dict(model['optimazer'])
            epoch_state = model["epoch"]
            print("load pre-train at epoch {}".format(epoch_state))
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    net = torch.nn.DataParallel(net, device_ids= list(eval(cfg.device_ids)) )

    criterion_Loss = torch.nn.L1Loss().to(cfg.device)    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 50, 55, 60, 65, 70, 75, 80], gamma=cfg.gamma)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 70, 80, 85, 90, 95, 100], gamma=cfg.gamma)
    scheduler._step_count = epoch_state
    loss_epoch = []
    loss_list = []

    for idx_epoch in range(epoch_state, cfg.n_epochs):
        for idx_iter, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
            out = net(data)
            #print(out.shape)
            #print(label.shape)
            loss = criterion_Loss(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            txtfile = open(savepath + cfg.tag + cfg.model_name + '_training.txt', 'a')
            txtfile.write(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())) + '\n')
            txtfile.close()
            writer.add_scalars('loss', {'trainset':float(np.array(loss_epoch).mean())}, idx_epoch )
            writer.flush()
            save_ckpt({
                'epoch': idx_epoch + 1,
                #'state_dict': net.state_dict(),
                'optimazer': optimizer.state_dict(),
                'state_dict': net.module.state_dict(),  # for torch.nn.DataParallel
                'loss': loss_list,},
                save_path=savepath, filename=cfg.model_name + '_' + str(cfg.upscale_factor) + 'xSR_' + str(cfg.angRes) +
                            'x' + str(cfg.angRes) + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')
            loss_epoch = []

        ''' evaluation '''
        with torch.no_grad():
            psnr_testset = []
            ssim_testset = []
            num_testset = []
            for index, test_name in enumerate(test_Names):
                test_loader = test_loaders[index]
                psnr_epoch_test, ssim_epoch_test, num_epoch_test = valid(test_loader, net)
                psnr_testset.append(psnr_epoch_test)
                ssim_testset.append(ssim_epoch_test)
                num_testset.append(num_epoch_test)

                print(time.ctime()[4:-5] + ' Valid----%15s,\t test Number---%d, PSNR---%f, SSIM---%f' % (test_name, num_epoch_test, psnr_epoch_test, ssim_epoch_test))
                txtfile = open(savepath + cfg.tag + cfg.model_name + '_training.txt', 'a')
                txtfile.write('Dataset----%10s,\t test Number---%d ,\t PSNR---%f,\t SSIM---%f\n' % (test_name, num_epoch_test, psnr_epoch_test, ssim_epoch_test))
                txtfile.close()

                # writer.add_scalars('psnr', {test_name:psnr_epoch_test} ,idx_epoch)
                # writer.add_scalars('ssim', {test_name:ssim_epoch_test} ,idx_epoch)
                # tensorboard
                
                pass
            psnr_avg = sum([psnr_testset[ii]*num_testset[ii] for ii in range(len(num_testset))]) / sum(num_testset)
            ssim_avg = sum([ssim_testset[ii]*num_testset[ii] for ii in range(len(num_testset))]) / sum(num_testset)

            txtfile = open(savepath + cfg.tag + cfg.model_name + '_training.txt', 'a')
            txtfile.write('Total testset,\t test Number---%d ,\t PSNR---%f,\t SSIM---%f\n' % (sum(num_testset), psnr_avg, ssim_avg))
            txtfile.close()
            writer.add_scalars('psnr', {'testset':psnr_avg}, idx_epoch)
            writer.add_scalars('ssim', {'testset':ssim_avg}, idx_epoch)

            writer.flush()
            pass


        scheduler.step()
        pass


def valid(test_loader, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angRes, w*angRes
        label = label.squeeze()

        uh, vw = data.shape
        h0, w0 = uh // cfg.angRes, vw // cfg.angRes
        subLFin = LFdivide(data, cfg.angRes, cfg.patchsize, cfg.stride)  # numU, numV, h*angRes, w*angRes
        numU, numV, H, W = subLFin.shape
        #subLFout = torch.zeros(numU, numV, cfg.angRes * cfg.patchsize*cfg.upscale_factor, cfg.angRes * cfg.patchsize*cfg.upscale_factor)

        minibatch = 2
        num_inference = numU*numV//minibatch
        tmp_in = subLFin.contiguous().view(numU*numV, subLFin.shape[2], subLFin.shape[3])
        
        

        with torch.no_grad():
            torch.cuda.empty_cache()
            out_lf = []
            for idx_inference in range(num_inference):
                tmp = tmp_in[idx_inference*minibatch:(idx_inference+1)*minibatch,:,:].unsqueeze(1)
                out_lf.append(net(tmp.to(cfg.device)))#
            if (numU*numV)%minibatch:
                tmp = tmp_in[(idx_inference+1)*minibatch:,:,:].unsqueeze(1)
                out_lf.append(net(tmp.to(cfg.device)))#
        out_lf = torch.cat(out_lf, 0)

        subLFout = out_lf.view(numU, numV, cfg.angRes * cfg.patchsize*cfg.upscale_factor, cfg.angRes * cfg.patchsize*cfg.upscale_factor)
        '''
        subLFout = torch.zeros(numU, numV, cfg.angRes * cfg.patchsize * cfg.upscale_factor, cfg.angRes * cfg.patchsize * cfg.upscale_factor)

        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    out = net(tmp.to(cfg.device))
                    subLFout[u, v, :, :] = out.squeeze()
        '''
        # print(idx_iter)
        # torch.cuda.empty_cache() # I dont know why, if uncomment this line, the code will stop here
        # print(idx_iter)
        outLF = LFintegrate(subLFout, cfg.angRes, cfg.patchsize * cfg.upscale_factor, cfg.stride * cfg.upscale_factor, h0 * cfg.upscale_factor, w0 * cfg.upscale_factor)

        psnr, ssim = cal_metrics(label, outLF, cfg.angRes)

        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())
    num_epoch_test = len(psnr_iter_test)

    return psnr_epoch_test, ssim_epoch_test, num_epoch_test


def save_ckpt(state, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))


def main(cfg):
    setup_seed(10)
    if not os.path.exists(savepath):
        try:
            os.mkdir(savepath)
        except:
            os.makedirs(savepath)


    os.system('cp -r ../code ' + savepath)
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir)
    train_loader = DataLoader(dataset=train_set, num_workers=32, batch_size=cfg.batch_size, shuffle=True)
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    train(cfg, train_loader, test_Names, test_Loaders)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False
if __name__ == '__main__':
    cfg = parse_args()
    global savepath, writer
    savepath = '../save/checkpoint_' + cfg.tag + '/'
    print(savepath)
    writer = SummaryWriter(os.path.join(savepath, 'tensorboard'))

    main(cfg)