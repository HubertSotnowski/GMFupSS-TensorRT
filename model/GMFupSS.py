import torch
import torch.nn.functional as F
from model.gmflow.gmflow import GMFlow
from model.MetricNet import MetricNet
from model.FusionNet import AnimeInterp as Fusionnet
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.flownet = GMFlow(num_scales=2, upsample_factor=4) # from gmflow
        self.metricnet = MetricNet() # from SoftmaxSplatting (trained)
        self.fusionnet = Fusionnet() # from wild animation interpolation (trained)
        self.flownet.eval()
        self.metricnet.eval()
        self.fusionnet.eval()
        rank = -1
        path="weights"
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            checkpoint = torch.load('{}/flownet.pkl'.format(path))
            weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
            self.flownet.load_state_dict(weights, strict=True)
            self.metricnet.load_state_dict(convert(torch.load('{}/metric.pkl'.format(path))))
            self.fusionnet.load_state_dict(convert(torch.load('{}/fusionnet.pkl'.format(path))))

    def forward(self, x2):
        x2=x2/2
        img1,img0=torch.split(x2,int(x2.shape[3]/2),dim=3)
        scale=1.0
        timestep=0.5
        torch.set_default_tensor_type(torch.FloatTensor)
        imgs = torch.cat((img0, img1), 1)
        img0_2 = F.interpolate(img0, scale_factor = 0.5, mode="bilinear", align_corners=False)
        img1_2 = F.interpolate(img1, scale_factor = 0.5, mode="bilinear", align_corners=False)
        if scale != 1.0:
            imgf0 = F.interpolate(imgs[:, :3], scale_factor = scale * 0.5, mode="bilinear", align_corners=False)
            imgf1 = F.interpolate(imgs[:, 3:6], scale_factor = scale * 0.5, mode="bilinear", align_corners=False)
        else:
            imgf0 = img0_2
            imgf1 = img1_2
        flow01 = self.flownet(imgf0, imgf1)
        flow10 = self.flownet(imgf1, imgf0)
        if scale != 1.0:
            flow01 = F.interpolate(flow01, scale_factor = 1. / scale, mode="bilinear", align_corners=False) / scale
            flow10 = F.interpolate(flow10, scale_factor = 1. / scale, mode="bilinear", align_corners=False) / scale
        metric0, metric1 = self.metricnet(img0_2, img1_2, flow01, flow10)

        #imgs = torch.cat((img0, img1), 1)
        img0 = F.interpolate(img0, scale_factor = 0.5, mode="bilinear", align_corners=False)
        img1 = F.interpolate(img1, scale_factor = 0.5, mode="bilinear", align_corners=False)
        F0t = timestep * flow01
        F1t = (1 - timestep) * flow10
        out = self.fusionnet(imgs, img0, img1, F0t, F1t, metric0, metric1)
        topad=int(x2.shape[3]/2)
        padding=torch.nn.ZeroPad2d([0, topad])
        out=padding(out)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out.shape)
        print(out)
        return out*2