from fastai.basics import *

############################################################################################

def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    '''Conv2d + ELU'''
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                         nn.ELU(inplace=True))

def ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    '''Conv2d + ELU + GroupNorm'''
    return nn.Sequential(ConvLayer(in_channels, out_channels, kernel_size, stride, padding),
                         nn.GroupNorm(num_groups=8, num_channels=out_channels, eps=1e-6))

############################################################################################

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_layers=4):
        super().__init__()
        self.first_layer = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.mid_layers = nn.ModuleList([ConvBlock(in_channels + i*out_channels, out_channels, kernel_size=3, stride=1, padding=1) 
                                         for i in range(1, nb_layers)])
        self.last_layer = ConvLayer(in_channels + nb_layers*out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        layers_concat = list()
        layers_concat.append(x)
        layers_concat.append(self.first_layer(x))
        
        for mid_layer in self.mid_layers:
            layers_concat.append(mid_layer(torch.cat(layers_concat, dim=1)))
            
        return self.last_layer(torch.cat(layers_concat, dim=1))

def AvgPoolDenseBlock(in_channels, out_channels, nb_layers=4):
    return nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
                         DenseBlock(in_channels, out_channels, nb_layers))

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convtrans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.elu = nn.ELU(inplace=True)
    
    def forward(self, x, x_enc):
        x = self.convtrans(x, output_size=x_enc.shape[-2:])
        x = self.elu(x)
        return torch.cat([x, x_enc], dim=1)

############################################################################################

class NetO(nn.Module):
    def __init__(self, num_in_frames=12, num_out_frames=12, active_node_feat = False, learn_incident=False):
        super().__init__()
        
        in_channels =9*num_in_frames+3*9+7+1
        if active_node_feat:
            in_channels +=1
        self.enc1 = DenseBlock(in_channels, 64, 4)
        self.enc2 = AvgPoolDenseBlock(64, 96, 4)
        self.enc3 = AvgPoolDenseBlock(96, 128, 4)
        self.enc4 = AvgPoolDenseBlock(128, 128, 4)
        self.enc5 = AvgPoolDenseBlock(128, 128, 4)
        self.enc6 = AvgPoolDenseBlock(128, 128, 4)
        self.enc7 = AvgPoolDenseBlock(128, 128, 4)
        self.enc8 = AvgPoolDenseBlock(128, 128, 4)
        
        self.bridge = ConvBlock(128, 128)
        
        self.dec7_1 = UpConvBlock(128, 128)
        self.dec7_2 = ConvBlock(128+128, 128)
        self.dec6_1 = UpConvBlock(128, 128)
        self.dec6_2 = ConvBlock(128+128, 128)
        self.dec5_1 = UpConvBlock(128, 128)
        self.dec5_2 = ConvBlock(128+128, 128)
        self.dec4_1 = UpConvBlock(128, 128)
        self.dec4_2 = ConvBlock(128+128, 128)
        self.dec3_1 = UpConvBlock(128, 128)
        self.dec3_2 = ConvBlock(128+128, 128)
        self.dec2_1 = UpConvBlock(128, 128)
        self.dec2_2 = ConvBlock(128+96, 128)
        self.dec1_1 = UpConvBlock(128, 128)
        self.dec1_2 = ConvBlock(128+64, 128)
        
        self.num_out_features = 8
        if learn_incident:
            self.num_out_features = 9
        
        self.out_1 = nn.Conv2d(128, self.num_out_features*num_out_frames, kernel_size=3, stride=1, padding=1)
        self.out_2 = nn.Sigmoid()
          
    def forward(self, x):
        
        x0 = x[:,:108,...]               # 4, 108, 495, 436
        s = x[:,108:,...]             # 4, 7, 495, 436
        #t = x[:,115,...].unsqueeze(1)    # 4, 1, 495, 436
        
        N, C, H, W = x0.shape            # 4, 108, 495, 436
        x0r = x0.reshape(N, 9, 12, H, W) # 4, 9, 12, 495, 436
        x0_mean = x0r.mean(dim=2)        # 4, 9, 495, 436
        x0_std = x0r.std(dim=2)
        x0_rng = x0r.max(dim=2).values - x0r.min(dim=2).values
        
        x = torch.cat([x0, x0_mean, x0_std, x0_rng, s], dim=1)
        x = x / 255.
        
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        
        x100 = self.bridge(x8)
        
        x107 = self.dec7_2(self.dec7_1(x100, x7))
        x106 = self.dec6_2(self.dec6_1(x107, x6))
        x105 = self.dec5_2(self.dec5_1(x106, x5))
        x104 = self.dec4_2(self.dec4_1(x105, x4))
        x103 = self.dec3_2(self.dec3_1(x104, x3))
        x102 = self.dec2_2(self.dec2_1(x103, x2))
        x101 = self.dec1_2(self.dec1_1(x102, x1))
        
        out = self.out_2(self.out_1(x101))*255.
        return out.view(N, self.num_out_features, -1, H, W)