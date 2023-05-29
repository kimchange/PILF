import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange
import math


class Net(nn.Module):
    def __init__(self, angRes, factor):
        super(Net, self).__init__()
        channels = 64
        self.channels = channels
        self.angRes = angRes
        self.factor = factor
        layer_num = 6


        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 8
        self.MHSA_params['dropout'] = 0.

        ##################### Initial Convolution #####################
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.conv_init = nn.Sequential(
              nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
              nn.LeakyReLU(0.2, inplace=True),
              CascadedBlocks(layer_num, channels, angRes),
            #   nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            #   nn.LeakyReLU(0.2, inplace=True),
              nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
              nn.LeakyReLU(0.2, inplace=True),
          )
        #self.Encoder = CascadeDisentgGroup(1, 4, channels, angRes)

        ################ Alternate AngTrans & SpaTrans ################
        self.altblock = self.make_layer(layer_num=layer_num)

        ####################### UP Sampling ###########################
        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels*self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
            # nn.Conv3d(channels, channels*self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.PixelShuffle(self.factor),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.Conv3d(channels, 1, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
        )

    def make_layer(self, layer_num):
        layers = []
        # layers.append(C42_Trans_serial(self.angRes, self.channels, self.MHSA_params, layer_num))
        for i in range(layer_num):
            layers.append(EPIX_Trans(self.angRes, self.channels, self.MHSA_params))
            # layers.append(CascadedBlocks(1, self.channels, self.angRes))
            # layers.append(C42_Trans_parallel(self.angRes, self.channels, self.MHSA_params))
        layers.append(nn.Conv3d(self.channels, self.channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, lr):
        # Bicubic

        b,u,v,n,h,w = lr.shape[0], self.angRes, self.angRes, self.angRes*self.angRes,lr.shape[2]//self.angRes, lr.shape[3]//self.angRes
        H,W = h*self.factor, w*self.factor
        lr = lr.reshape(b, lr.shape[1], u, h, v, w)
        lr = lr.permute(0,1,2,4,3,5)
        lr = lr.reshape(b, lr.shape[1], n, h, w) #b,1,n,h,w

        # Bicubic
        lr_upscale = F.interpolate(lr.reshape(b* lr.shape[1]* n, 1, h, w),scale_factor= self.factor, mode='bicubic', align_corners=False).reshape(b,1,n,H,W)



        # Initial Convolution
        buffer_init = self.conv_init0(lr)
        buffer = self.conv_init(buffer_init)+buffer_init


        # EPIXTrans
        buffer = self.altblock(buffer) + buffer


        buffer = buffer.permute(0, 2, 1, 3, 4).reshape(b*n, self.channels, h, w)

        buffer = self.upsampling(buffer).reshape(b,n, 1, H, W).permute(0, 2, 1, 3, 4)
        out = buffer + lr_upscale
        

        out = out.reshape(b,1,u,v,H,W).permute(0,1,2,4,3,5).reshape(b,1,u*H,v*W)
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

    def gen_mask(self, h: int, w: int, maxdisp: int=2):
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
        S_ch, A_ch, E_ch, D_ch  = ch, ch//4, ch//2, ch//2
        self.angRes = angRes
        self.spaconv  = SpatialConv(ch)
        self.angconv  = AngularConv(ch, angRes, A_ch)
        self.epiconv = EPiConv(ch, angRes, E_ch)
        self.dpiconv = EPiConv(ch, angRes, D_ch)
        self.fuse = nn.Sequential(
                nn.Conv3d(in_channels = S_ch+A_ch+E_ch+E_ch+D_ch+D_ch, out_channels = ch, kernel_size = 1, stride = 1, padding = 0, dilation=1),
                nn.LeakyReLU(0.2, inplace=True),
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
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv3d(in_channels = ch, out_channels = ch, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=(1,1,1)),
                    nn.LeakyReLU(0.2, inplace=True)
                    )

    def forward(self,fm):

        return self.spaconv_s(fm) 



class AngularConv(nn.Module):
    def __init__(self, ch, angRes, AngChannel):
        super(AngularConv, self).__init__()
        self.angconv = nn.Sequential(
            nn.Conv3d(ch*angRes*angRes, AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(AngChannel, AngChannel * angRes * angRes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
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
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv3d(EPIChannel, angRes * EPIChannel, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False), # ksize maybe (1,1,angRes//2*2+1) ?
                    nn.LeakyReLU(0.2, inplace=True),
                    # PixelShuffle1D(angRes),
                    )
        #self.an = angRes

    def forward(self,fm):
        b, c, u, v, h, w = fm.shape

        epih_in = fm.permute(0, 1, 2, 4, 3, 5).reshape(b,c,u*h,v,w)
        epih_out = self.epiconv(epih_in) # (b,self.epi_ch*v, u*h, 1, w)
        epih_out = epih_out.reshape(b,self.epi_ch,v,u,h,w).permute(0,1,3,2,4,5).reshape(b, self.epi_ch,u*v,h,w)
        return epih_out


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
        # x = x.permute(0, 2, 1, 3, 4)
        # b, n, c, h, w = x.shape     
        buffer = x
        for i in range(self.n_blocks):
            buffer = self.body[i](buffer)        
        # buffer = self.conv(buffer.contiguous().view(b*n, c, h, w))
        buffer = self.conv(buffer) + x
        # buffer = buffer.contiguous().view(b,n, c, h, w) + x
        return buffer# buffer.permute(0, 2, 1, 3, 4)

class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR):
        loss = self.criterion_Loss(SR, HR)

        return loss


if __name__ == "__main__":
    net = Net(5, 4).cuda()
    from thop import profile
    input = torch.randn(1, 1, 160, 160).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))
