####### CNN ENCODER

class Encoder(nn.Module):
    
    def __init__(self, backbone, num_channels = 3, pretrained = True):
        super().__init__()
        self.backbone = timm.create_model(model_name = backbone, 
                                          in_chans   = num_channels, 
                                          pretrained = pretrained)
        
    def forward(self, x):
        bs = x.size(0)
        features = self.backbone.forward_features(x)
        features = features.permute(0, 2, 3, 1)
        return features