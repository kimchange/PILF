import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

class Net(nn.Module):
    def __init__(self, angular_in, angular_out):
        super(Net, self).__init__()
        ngroup, nblock, channel = 1, 6, 64
        self.channel= channel
        self.angRes = angular_in
        self.angRes_out = angular_out
        layer_num = 8
        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 8
        self.MHSA_params['dropout'] = 0.
        self.FeaExtract = nn.Sequential(
            nn.Conv3d(1, channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.DeepFeaExt = CascadedBlocks(layer_num, channel, angular_in)
        self.UpSample = nn.Sequential(
            nn.Conv3d(channel*angular_in*angular_in, channel//4*angular_in*angular_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channel//4*angular_in*angular_in, channel//4 * angular_out * angular_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channel//4*angular_out*angular_out, angular_out * angular_out, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.altblock = self.make_layer(layer_num=layer_num)
    def make_layer(self, layer_num):
        layers = []
        # layers.append(C42_Trans_serial(self.angRes, self.channels, self.MHSA_params, layer_num))
        for i in range(layer_num):
            layers.append(EPIX_Trans(self.angRes, self.channel, self.MHSA_params))
            # layers.append(CascadedBlocks(1, self.channels, self.angRes))
            # layers.append(C42_Trans_parallel(self.angRes, self.channels, self.MHSA_params))
        layers.append(nn.Conv3d(self.channel, self.channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False))
        return nn.Sequential(*layers)


    def forward(self, x):
        x_mv = LFsplit(x, self.angRes)
        buffer = self.FeaExtract(x_mv) 
        buffer = self.altblock(self.DeepFeaExt(buffer))
        b,c,n,h,w = buffer.shape

        buffer = buffer.contiguous().view(b,c*n,1,h,w)
        buffer = self.UpSample(buffer).view(b,1,self.angRes_out*self.angRes_out,h,w) # n == angRes * angRes
        out = FormOutput(buffer)
        return out


class InitFeaExtract(nn.Module):
    def __init__(self, channel):
        super(InitFeaExtract, self).__init__()
        self.FEconv = nn.Sequential(
            nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        b, n, _, h, w = x.shape
        x = x.contiguous().view(b*n, -1, h, w)
        buffer = self.FEconv(x)
        _, c, h, w = buffer.shape
        buffer = buffer.unsqueeze(1).contiguous().view(b, -1, c, h, w)#.permute(0,2,1,3,4)  # buffer_sv:  B, N, C, H, W

        return buffer
class Upsample(nn.Module):
    def __init__(self, channel, angular_in, angular_out):
        super(Upsample, self).__init__()
        self.an = angular_in
        self.an_out = angular_out
        self.angconv = nn.Sequential(
                        )
        self.upsp = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=angular_in, stride=angular_in, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel* angular_out * angular_out, kernel_size=1, padding=0),
            nn.PixelShuffle(angular_out),
            nn.Conv2d(channel, 1, kernel_size=3, padding=1))

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.contiguous().view(b, n, c, h*w)
        x = torch.transpose(x, 1, 3)
        x = x.contiguous().view(b*h*w, c, self.an, self.an)
        up_in = self.angconv(x)

        out = self.upsp(up_in)

        out = out.view(b,h*w,-1,self.an_out*self.an_out)
        out = torch.transpose(out,1,3)
        out = out.contiguous().view(b, self.an_out*self.an_out, -1, h, w)   #[N*81,c,h,w]
        return out
class EpiXTrans(nn.Module):
    def __init__(self, channels, emb_dim, MHSA_params):
        super(EpiXTrans, self).__init__()
        self.emb_dim = emb_dim
        self.linear_in = nn.Linear(channels, emb_dim, bias=False)
        self.norm = nn.LayerNorm(emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim, 
                                               MHSA_params['num_heads'], 
                                               MHSA_params['dropout'], 
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(emb_dim*2, emb_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )
        self.linear_out = nn.Linear(emb_dim, channels, bias=False)

    ######### very important!!!
    def gen_mask(self, h: int, w: int, maxdisp: int=6):    # when 30 Scenes Reflective Occlusion
    # def gen_mask(self, h: int, w: int, maxdisp: int=18):  # when HCI data
        attn_mask = torch.zeros([h, w, h, w])
        # k_h_left = k_h // 2
        # k_h_right = k_h - k_h_left
        # k_w_left = k_w // 2
        # k_w_right = k_w - k_w_left
        [ii,jj] = torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')

        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[(ii-i).abs() * maxdisp >= (jj-j).abs()] = 1
                # temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        # attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.reshape(h*w, h*w)
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, buffer):
        # [_, _, n, v, w] = buffer.size()
        # b, c, u, h, v, w = buffer.shape
        b, c, u, v, h, w = buffer.shape
        # attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device)
        attn_mask = self.gen_mask(v, w, ).to(buffer.device)

        # epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        epi_token = buffer.permute(3,5,0,2,4,1).reshape(v*w, b*u*h, c)
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        # buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)
        buffer = epi_token.reshape(v,w, b,u,h, c).permute(2, 5, 3, 0, 4, 1).reshape(b, c, u, v, h, w)

        return buffer



class EPIX_Trans(nn.Module):
    def __init__(self, angRes, channels, MHSA_params):
        super(EPIX_Trans, self).__init__()
        self.angRes = angRes

        self.epi_trans = EpiXTrans(channels, channels*2, MHSA_params)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )
        # self.conv_2 = nn.Sequential(
        #     nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        # )
    def forward(self, x):
        
        # [_, _, _, h, w] = x.size()
        b, c, n, h, w = x.size()
        
        u, v = self.angRes, self.angRes


        shortcut = x
        # self.epi_trans.mask_field = [self.angRes, self.angRes]
        # self.dpi_trans.mask_field = [self.angRes, self.angRes]
        # self.epi_trans.mask_field = [self.angRes * 2, 11]
        # self.dpi_trans.mask_field = [self.angRes * 2, 11]

        # EPI uh
        buffer = x.reshape(b,c,u,v,h,w).permute(0,1,3,2,5,4)
        buffer = self.conv_1( self.epi_trans(buffer).permute(0,1,3,2,5,4).reshape(b,c,n,h,w) ) + shortcut
        # shortcut = buffer

        # # DPI uw
        # buffer = buffer.reshape(b,c,u,v,h,w).permute(0,1,3,2,4,5)
        # buffer = self.conv_2( self.dpi_trans(buffer).permute(0,1,3,2,4,5).reshape(b,c,n,h,w) ) + shortcut
        # shortcut = buffer

        # # Ang uv
        # buffer = buffer.reshape(b,c,u,v,h,w)
        # buffer = self.conv_3( self.ang_trans(buffer).reshape(b,c,n,h,w) ) + shortcut
        # shortcut = buffer
        

        # EPI vw
        buffer = buffer.reshape(b,c,u,v,h,w)
        buffer = self.conv_1( self.epi_trans(buffer).reshape(b,c,n,h,w) ) + shortcut
        # shortcut = buffer

        
        # # DPI vh
        # buffer = buffer.reshape(b,c,u,v,h,w).permute(0,1,2,3,5,4)
        # buffer = self.conv_2( self.dpi_trans(buffer).permute(0,1,2,3,5,4).reshape(b,c,n,h,w) ) + shortcut
        # shortcut = buffer

        # # Spa hw
        # buffer = buffer.reshape(b,c,u,v,h,w)
        # buffer = self.conv_4( self.spa_trans(buffer).reshape(b,c,n,h,w)) + shortcut

        return buffer

class C42_Conv(nn.Module):
    def __init__(self, ch, angRes):
        super(C42_Conv, self).__init__()
                
        self.relu = nn.ReLU(inplace=True)
        S_ch, A_ch, E_ch, D_ch  = ch, ch, ch//2, ch//2
        self.angRes = angRes
        self.spaconv  = SpatialConv(ch)
        self.angconv  = AngularConv(ch, angRes, A_ch)
        self.epiconv = EPiConv(ch, angRes, E_ch)
        self.dpiconv = EPiConv(ch, angRes, D_ch)
        self.fuse = nn.Sequential(
                nn.Conv3d(in_channels = S_ch+A_ch+E_ch+E_ch+D_ch+D_ch, out_channels = ch, kernel_size = 1, stride = 1, padding = 0, dilation=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv3d(ch, ch, kernel_size = (1,3,3), stride = 1, padding = (0,1,1),dilation=1))
    
    def forward(self,x):
        # b, n, c, h, w = x.shape
        b, c, n, h, w = x.shape
        an = int(math.sqrt(n))
        s_out = self.spaconv(x)
        a_out = self.angconv(x)

        epih_in = x.contiguous().view(b, c, an, an, h, w) # b,c,u,v,h,w
        epih_out = self.epiconv(epih_in)

        # epiv_in = epih_in.permute(0,2,1,3,5,4)
        epiv_in = epih_in.permute(0,1,3,2,5,4) # b,c,v,u,w,h
        epiv_out = self.epiconv(epiv_in).reshape(b, -1, an, an, w, h).permute(0,1,3,2,5,4).reshape(b, -1, n, h, w)
        
        dpih_in = epih_in.permute(0,1,3,2,4,5) # b,c,v,u,h,w
        dpih_out = self.dpiconv(dpih_in).reshape(b, -1, an, an, w, h).permute(0,1,3,2,4,5).reshape(b, -1, n, h, w)

        dpiv_in = epih_in.permute(0,1,2,3,5,4) # b,c,u,v,w,h
        dpiv_out = self.dpiconv(dpiv_in).reshape(b, -1, an, an, w, h).permute(0,1,2,3,5,4).reshape(b, -1, n, h, w)
        

        out = torch.cat((s_out, a_out, epih_out, epiv_out, dpih_out, dpiv_out), 1)
        out = self.fuse(out)

        return out + x# out.contiguous().view(b,n,c,h,w) + x


class SpatialConv(nn.Module):
    def __init__(self, ch):
        super(SpatialConv, self).__init__()
        self.spaconv_s = nn.Sequential(
                    nn.Conv3d(in_channels = ch, out_channels = ch, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=(1,1,1)),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    nn.Conv3d(in_channels = ch, out_channels = ch, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=(1,1,1)),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self,fm):

        return self.spaconv_s(fm) 



class AngularConv(nn.Module):
    def __init__(self, ch, angRes, AngChannel):
        super(AngularConv, self).__init__()
        self.angconv = nn.Sequential(
            nn.Conv3d(ch*angRes*angRes, AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(AngChannel, AngChannel * angRes * angRes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.PixelShuffle(angRes)
        )
        # self.an = angRes

    def forward(self,fm):
        b, c, n, h, w = fm.shape
        a_in = fm.contiguous().view(b,c*n,1,h,w)
        out = self.angconv(a_in).view(b,-1,n,h,w) # n == angRes * angRes
        return out

class EPiConv(nn.Module):
    def __init__(self, ch, angRes, EPIChannel):
        super(EPiConv, self).__init__()
        self.epi_ch = EPIChannel
        self.epiconv = nn.Sequential(
                    nn.Conv3d(ch, EPIChannel, kernel_size=(1, angRes, angRes//2*2+1), stride=1, padding=(0, 0, angRes//2), bias=False),
                    nn.LeakyReLU(0.1, True),
                    nn.Conv3d(EPIChannel, angRes * EPIChannel, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False), # ksize maybe (1,1,angRes//2*2+1) ?
                    nn.LeakyReLU(0.1, True),
                    # PixelShuffle1D(angRes),
                    )
        #self.an = angRes

    def forward(self,fm):
        b, c, u, v, h, w = fm.shape

        epih_in = fm.permute(0, 1, 2, 4, 3, 5).reshape(b,c,u*h,v,w)
        epih_out = self.epiconv(epih_in) # (b,self.epi_ch*v, u*h, 1, w)
        epih_out = epih_out.reshape(b,self.epi_ch,v,u,h,w).permute(0,1,3,2,4,5).reshape(b, self.epi_ch,u*v,h,w)
        return epih_out


class PixelShuffle1D(nn.Module):
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor

        return x.view(b, c, h * self.factor, w)


class CascadedBlocks(nn.Module):
    '''
    Hierarchical feature fusion
    '''
    def __init__(self, n_blocks, channel, angRes):
        super(CascadedBlocks, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(C42_Conv(channel, angRes))
        self.body = nn.Sequential(*body)
        # self.conv = nn.Conv2d(channel, channel, kernel_size = (3,3), stride = 1, padding = 1, dilation=1)
        self.conv = nn.Conv3d(channel, channel, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=1)

    def forward(self, x):
        buffer = x
        for i in range(self.n_blocks):
            buffer = self.body[i](buffer)        
        buffer = self.conv(buffer) + x
        return buffer

class CascadeC42Group(nn.Module):
    def __init__(self, n_group, n_block, channels, angRes):
        super(CascadeC42Group, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(CascadedBlocks(n_block, channels, angRes))
        self.Group = nn.Sequential(*Groups)      
        self.conv = nn.Conv3d(channels, channels, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=1)

    def forward(self, x):  
        buffer = x
        for i in range(self.n_group):
            buffer = self.Group[i](buffer)            
        buffer = self.conv(buffer)
        return buffer + x
def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st.permute(0,2,1,3,4)


def FormOutput(x_sv):
    x_sv = x_sv.permute(0,2,1,3,4)
    b, n, c, h, w = x_sv.shape
    angRes = int(math.sqrt(n+1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(x_sv[:, kk, :, :, :])
            kk = kk+1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    net = Net(2, 7).cuda()
    from thop import profile
    ##### get input index ######         
    input = torch.randn(1, 1, 128, 128).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.4fM' % (total / 1e6))
    print('   Number of FLOPs: %.4fG' % (flops / 1e9))
