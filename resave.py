import os
import torch
import torch.nn as nn
from models import ModelBuilder, SegmentationModule
import torchvision.models as models
from models import resnet

def resave():
    arch_encoder = "resnet50"
    arch_decoder = "ppm_bilinear_deepsup"

    # weights_encoder="./ckpt/MacNetV2_mobilev2_scse_ppm/encoder_epoch_19.pth"
    weights_encoder=""
    weights_decoder=""

    fc_dim=2048

    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=arch_encoder,
        fc_dim=fc_dim,
        weights=weights_encoder,
        use_softmax=True)
    net_decoder = builder.build_decoder(
        arch=arch_decoder,
        fc_dim=fc_dim,
        num_class=14,
        weights=weights_decoder)

    #net_encoder = resnet.__dict__['resnet50'](pretrained=False)
    net_encoder = resnet.resnet50(pretrained=False)

    print(net_encoder)

    save_path = "./ckpt/MacNetV2_mobilev2_scse_ppm/encoder_full_epoch_19.pth"
    torch.save(net_encoder, save_path)



if __name__ == '__main__':
    resave()