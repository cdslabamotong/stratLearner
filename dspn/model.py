
import torch.nn as nn
import torch.nn.init as init
from dspn import DSPN



def build_net(args):
    set_channels = args.vNum
    set_size = args.set_size

    output_channels = 256

    set_encoder_dim=1024
    input_encoder_dim = 256

    inner_lr = args.inner_lr
    iters = args.iters

    set_encoder_class = globals()[args.encoder]
    set_decoder_class = globals()[args.decoder]
    
    set_encoder = set_encoder_class(set_channels, output_channels, set_size, set_encoder_dim)

    if set_decoder_class == DSPN:
        set_decoder = DSPN(
            set_encoder, set_channels, set_size, iters, inner_lr,args.batch_size
        )
    else:
        pass
    
    input_encoder = MLPEncoderInput(set_channels, output_channels, set_size, input_encoder_dim)
    
    net = Net_test(
        input_encoder=input_encoder, set_encoder=set_encoder, set_decoder=set_decoder
    )
    return net        
    
class Net_test(nn.Module):
    def __init__(self, set_encoder, set_decoder, input_encoder=None):
        """
        In the auto-encoder setting, don't pass an input_encoder because the target set and mask is
        assumed to be the input.
        In the general prediction setting, must pass all three.
        """
        super().__init__()
        self.set_encoder = set_encoder
        self.input_encoder = input_encoder
        self.set_decoder = set_decoder
        
        for m in self.modules():
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
            ):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, target_set, max_set_size):
        if self.input_encoder is None:
            # auto-encoder, ignore input and use target set and mask as input instead
            print("HERE 1")
            #latent_repr = self.set_encoder(input, target_mask)
            #print("HERE 2")

            #target_repr = self.set_encoder(target_set, target_mask)
        else:
            #print("HERE 3")
            # set prediction, use proper input_encoder
            latent_repr = self.input_encoder(input)
            # note that target repr is only used for loss computation in training
            # during inference, knowledge about the target is not needed
            target_repr = self.set_encoder(target_set)
        #print("target_repr.shape {}".format(target_repr.shape))
        predicted_set = self.set_decoder(latent_repr, max_set_size)
        return predicted_set, (latent_repr, target_repr)


############
# Encoders #
############

    
class MLPEncoderInput(nn.Module):
    def __init__(self, input_channels, output_channels, set_size, dim):
        super().__init__()
        self.output_channels = output_channels
        self.set_size = set_size
        self.model = nn.Sequential(
            nn.Linear(input_channels, dim), 
            nn.ReLU(),

            nn.Linear(dim, dim),
            nn.ReLU(),
            
            nn.Linear(dim, 256),
            nn.ReLU(),

            nn.Linear(256, output_channels),
        )

    def forward(self, x, mask=None):
        x1=x.sum(2)
        x = self.model(x1)
        return x

class MLPEncoder(nn.Module):
    def __init__(self, input_channels, output_channels, set_size, dim):
        super().__init__()
        self.output_channels = output_channels
        self.set_size = set_size
        self.model = nn.Sequential(
            nn.Linear(input_channels, dim), 
            nn.ReLU(),
            #nn.Linear(dim, dim),
            #nn.ReLU(),
            nn.Linear(dim, output_channels),
        )

    def forward(self, x, mask=None):
        x1=x.sum(2)
        x = self.model(x1)
        return x




############
# Decoders #
############


