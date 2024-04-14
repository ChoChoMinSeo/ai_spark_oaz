from efficientnet_pytorch import EfficientNet
from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock
import torch.nn as nn

class CWUnet(nn.Module):
    def __init__(self):
        super(CWUnet, self).__init__()     
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')  
        self.pmap_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3),stride=(2,2), bias=False, padding=(1,1)),
            nn.BatchNorm2d(32),
        )
        self.backbone._conv_stem = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3,3),stride=(2,2), bias=False, padding=(1,1))

        self.stage1 = nn.Sequential(
            self.backbone._conv_stem,
            self.backbone._bn0,
        )
        self.stage2 = nn.Sequential(
            self.backbone._blocks[0],
            self.backbone._blocks[1],
            self.backbone._blocks[2]
        )
        self.stage3 = nn.Sequential(
            self.backbone._blocks[3],
            self.backbone._blocks[4],
        )
        self.stage4 = nn.Sequential(
            self.backbone._blocks[5],
            self.backbone._blocks[6],
            self.backbone._blocks[7],
            self.backbone._blocks[8]
        )
        self.stage5 = nn.Sequential(
            self.backbone._blocks[9],
            self.backbone._blocks[10],
            self.backbone._blocks[11],
            self.backbone._blocks[12],
            self.backbone._blocks[13],
            self.backbone._blocks[14],
            self.backbone._blocks[15],
        )
        
        self.decoder1 = DecoderBlock(320,112,256)
        self.decoder2 = DecoderBlock(256,40,128)
        self.decoder3 = DecoderBlock(128,24,64)
        self.decoder4 = DecoderBlock(64,32,32)
        self.decoder5 = DecoderBlock(32,0,16)
        
        self.segmentationHead = nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    def forward(self, x):
        # B C H W
        p_map = x[:,0:1,:,:]
        x = x[:,1:,:,:]
        x1 = self.stage1(x)
        p_map = self.pmap_encoder(p_map)
        x1 += p_map # 32
        
        x2 = self.stage2(x1) # 24
        x3 = self.stage3(x2) # 40
        x4 = self.stage4(x3) # 112
        x5 = self.stage5(x4) # 320
        
        res = self.decoder1(x5,x4)
        res = self.decoder2(res,x3)
        res = self.decoder3(res,x2)
        res = self.decoder4(res,x1)
        res = self.decoder5(res)
        
        res = self.segmentationHead(res)
        
        return res
import torchsummary
torchsummary.summary(CWUnet().to('cuda'),(3,256,256))
