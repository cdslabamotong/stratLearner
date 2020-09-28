# -*- coding: utf-8 -*-
  
import torch
import torch.nn as nn
import torch.nn.functional as F



class DSPN(nn.Module):

    def __init__(self, encoder,  set_channels,max_set_size,  iters, lr, batch_size):
        """
        encoder: Set encoder module that takes a set as input and returns a representation thereof.
            It should have a forward function that takes two arguments:
            - a set: FloatTensor of size (batch_size, input_channels, maximum_set_size). Each set
            should be padded to the same maximum size with 0s, even across batches.
            - a mask: FloatTensor of size (batch_size, maximum_set_size). This should take the value 1
            if the corresponding element is present and 0 if not.
        channels: Number of channels of the set to predict.
        max_set_size: Maximum size of the set.
        iter: Number of iterations to run the DSPN algorithm for.
        lr: Learning rate of inner gradient descent in DSPN.
        """
        super().__init__()
        self.encoder = encoder
        self.iters = iters
        self.lr = lr
        self.batch_size = batch_size
        self.set_channels=set_channels

        self.starting_set = nn.Parameter(torch.rand(1, set_channels, max_set_size))
        
        #self.starting_mask = nn.Parameter(0.5 * torch.ones(1, max_set_size))
        #self.linear = nn.Linear(set_channels*max_set_size, max_set_size)

    def forward(self, target_repr, max_set_size):
        """
        Conceptually, DSPN simply turns the target_repr feature vector into a set.
        target_repr: Representation that the predicted set should match. FloatTensor of size (batch_size, repr_channels).
        Note that repr_channels can be different from self.channels.
        This can come from a set processed with the same encoder as self.encoder (auto-encoder), or a different
        input completely (normal supervised learning), such as an image encoded into a feature vector.
        """
        # copy same initial set over batch
        
        current_set = self.starting_set.expand(
            target_repr.size(0), *self.starting_set.size()[1:]
        ).detach().cpu()
        #current_set = self.starting_set
        #print(current_set.shape)
        #current_mask = self.starting_mask.expand(
        #    target_repr.size(0), self.starting_mask.size()[1]
        #)
        
        #current_set = self.starting_set 
        # make sure mask is valid
        #current_mask = current_mask.clamp(min=0, max=1)
        
        # info used for loss computation
        intermediate_sets = [current_set]
        #intermediate_masks = [current_mask]
        # info used for debugging
        repr_losses = []
        grad_norms = []
        
                    #self.starting_set.requires_grad = True
        for i in range(self.iters):
            # regardless of grad setting in train or eval, each iteration requires torch.autograd.grad to be used
            with torch.enable_grad():
                if not  self.training or True:
                    current_set.requires_grad = True

                predicted_repr = self.encoder(current_set)
                repr_loss = F.smooth_l1_loss(
                    predicted_repr, target_repr, reduction="mean"
                )
                
                # change to make to set and masks to improve the representation
                set_grad = torch.autograd.grad(
                    inputs=[current_set],
                    outputs=repr_loss,
                    only_inputs=True,
                    create_graph=True,
                )[0]


            
            current_set = current_set - self.lr * set_grad
            
            
            current_set = current_set.detach().cpu()

            repr_loss = repr_loss.detach().cpu()
            set_grad = set_grad.detach().cpu()

                
            # keep track of intermediates
            #print(current_set.shape)
            #print(current_set.sum(2).shape)
            intermediate_sets.append(current_set.sum(2))
            #intermediate_masks.append(current_mask)
            repr_losses.append(repr_loss)
            grad_norms.append(set_grad.norm())
        
        '''
        for i in range(len(intermediate_sets)):
            intermediate_sets[i] = self.linear(intermediate_sets[i].view(intermediate_sets[i].shape[0], -1))
            #intermediate_sets[i] = intermediate_sets[i].div_(torch.norm(intermediate_sets[i],2))
            intermediate_sets[i] = F.normalize(intermediate_sets[i], dim=1)

        '''
        return intermediate_sets, None, repr_losses, grad_norms
      
      
