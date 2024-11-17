#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import piq
import os

cudnn.benchmark = True


# VGGnet only
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
        self.stop_at = len(extracted_layers)

    def forward(self, x):
        outputs = []
        count = 0
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                count += 1
                outputs.append(x)
                if count == self.stop_at:
                    return outputs
        return outputs


# default 2
def create_perceptual_loss2(value, img_tensor, model, gamma=10000,
                            scalar=1, layers=['9', '19', '29', '38', '48']):
    img_tensor = img_tensor.detach()
    featex = FeatureExtractor(model.features, layers)
    target = model(img_tensor).argmax().detach()
    init_feat = featex(img_tensor)
    L2 = torch.nn.MSELoss()

    def loss(img):
        # perceptual loss
        out = 0
        for f, g in zip(init_feat, featex(img)):
            out += L2(f, g)
        out *= gamma
        out += L2(img, img_tensor)*scalar

        # l2 loss of response
        response = model(img)
        others = torch.max(response[0, :target].max(),
                           response[0, target + 1:].max())
        lab = response[0, target]
        return out + (lab - others - value)**2
    return loss


def _calculate_brisque_loss(image, original):
    # Transform to [0,1]
    image = (image - image.min()) / (image.max() - image.min())
    original = (original - original.min()) / (original.max() - original.min())

    loss = piq.BRISQUELoss(data_range=1.0, reduction='none')
    return loss(image) - loss(original)

def _reweigh_losses(losses):
    # Losses is a torch tensor. 
    norm = torch.norm(losses, p=2)
    weights = torch.pow(losses, 2) / norm
    # Want to reweigh by 1/weights, but need to avoid division by zero
    reweighted = torch.pow(weights + 1e-6, -0.5) * losses

    return reweighted

def _normalize_loss(losses):
    # Losses is a torch tensor. 
    norm = torch.norm(losses, p=2)
    reweighed = losses / norm
    # If some loss is negative, transform to positive
    idx = reweighed < 0
    reweighed[idx] = 1 + reweighed[idx]
    return reweighed

    
def create_perceptual_loss_dino(value, pre_dino_fn, img_tensor, img_dino_tensor, model, dino_model, gamma=10000, scalar=1, dino_scalar=10, target_scalar=100, nat_scale=100, layers=['9', '19', '29', '38', '48'], targetID=None):
    device = img_tensor.device
    img_tensor = img_tensor.detach()
    featex = FeatureExtractor(model.features, layers)
    target = model(img_tensor).argmax().detach() if targetID is None else targetID
    init_feat = featex(img_tensor)
    print(f"Target: {target}")

    with torch.no_grad():
        dino_features = dino_model.forward_features(img_dino_tensor)
    clstoken = dino_features['x_norm_clstoken']

    L2 = torch.nn.MSELoss()
    if targetID is not None:
        baseline = model(img_tensor)[0, targetID].abs()


    def loss(img):
        # perceptual loss
        perceptual_loss = 0
        for f, g in zip(init_feat, featex(img)):
            perceptual_loss += L2(f, g)
        perceptual_loss *= gamma

        prior_loss = L2(img, img_tensor)*scalar

        # dinov2 loss
        img_dino = pre_dino_fn(img)
        dino_features = dino_model.forward_features(img_dino)
        clstoken_dino = dino_features['x_norm_clstoken']
        dino_loss = dino_scalar / torch.pow(L2(clstoken, clstoken_dino) + 0.5, 0.5)

        brisque_loss = _calculate_brisque_loss(img, img_tensor).item() * nat_scale
        brisque_loss = torch.tensor(brisque_loss, device=device)

        # l2 loss of response
        response = model(img)
        label_loss = 0
        if targetID is None:
            others = torch.max(response[0, :target].max(),
                            response[0, target + 1:].max())
            lab = response[0, target]
            label_loss = (lab - others - value)**2
        else:
            # Perturb img to maximize response of targetID
            lab = response[0, targetID]
            label_loss = target_scalar / (lab + baseline + 1)
            # print(f"Label type: {type(label_loss)}, Label shape: {label_loss.shape}")


        losses = [perceptual_loss, prior_loss, dino_loss, brisque_loss, label_loss]
        # reweighted_losses = _reweigh_losses(torch.stack(losses))
        # losses = _normalize_loss(torch.stack(losses))
        return losses

    return loss

# multi-class chris
def create_perceptual_loss_multiclass(value, img_tensor, model, gamma=10000,
                                      scalar=1,
                                      layers=['9', '19', '29', '38', '48'],
                                      targetID=None):
    img_tensor = img_tensor.detach()
    featex = FeatureExtractor(model.features, layers)
    target = targetID
    init_feat = featex(img_tensor)
    L2 = torch.nn.MSELoss()
    r = model(img_tensor)

    def loss(img):
        # perceptual loss
        out = 0
        for f, g in zip(init_feat, featex(img)):
            out += L2(f, g)
        out *= gamma
        out += L2(img, img_tensor)*scalar
        response = model(img)
        lab = response[0, target].max()
        return out + (lab - value)**2
    return loss


def find_direction(
        loss,
        factual, # image_variable
        mins=False,
        maxs=False,
        iterations=200,
        save_losses=False
        ):

    direction = factual.data.clone()
    direction.requires_grad = True
    # set up loss
    optimizer = torch.optim.LBFGS([direction, ], max_iter=iterations)
    losses = []

    def closure():
        # Clamp_ doesn't take vectors
        # This can constrain the output if desired
        if mins is not False:
            min_mask = direction.data < mins
            direction.data[min_mask] = mins[min_mask]
        if maxs is not False:
            max_mask = direction.data > maxs
            direction.data[max_mask] = mins[max_mask]
        l = loss(direction) # x'

        if save_losses:
            losses.append(l.item())
        if len(losses) % 10 == 0:
            print(f"Iteration: {len(losses)}, Loss: {l.item()}")
        optimizer.zero_grad()
        l.backward(retain_graph=True)
        return l
    

    optimizer.step(closure)

    print(f"Iterations: {len(losses)}")
    if save_losses:
        return direction, losses
    
    return direction

def dino_find_direction(
        loss,
        factual, # image_variable
        mins=False,
        maxs=False,
        iterations=50,
        epochs=10,
        save_losses=False,
        out_dir='results'
        ):

    direction = factual.data.clone()
    direction.requires_grad = True
    # set up loss
    optimizer = torch.optim.LBFGS([direction, ], max_iter=iterations, lr=1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    losses = {}

    for epoch in range(epochs):
        def closure():
            # Clamp_ doesn't take vectors
            # This can constrain the output if desired
            if mins is not False:
                min_mask = direction.data < mins
                direction.data[min_mask] = mins[min_mask]
            if maxs is not False:
                max_mask = direction.data > maxs
                direction.data[max_mask] = mins[max_mask]
            perceptual_loss, prior_loss, dino_loss, brisque_loss, label_loss = loss(direction) # x'

            l = perceptual_loss + prior_loss + dino_loss + brisque_loss + label_loss

            if save_losses:
                # Save list of losses. So check if key exists
                if 'perceptual_loss' not in losses:
                    losses['perceptual_loss'] = []
                    losses['prior_loss'] = []
                    losses['dino_loss'] = []
                    losses['brisque_loss'] = []
                    losses['label_loss'] = []
                    losses['total_loss'] = []

                losses['perceptual_loss'].append(perceptual_loss.item())
                losses['prior_loss'].append(prior_loss.item())
                losses['dino_loss'].append(dino_loss.item())
                losses['brisque_loss'].append(brisque_loss.item())
                losses['label_loss'].append(label_loss.item())
                losses['total_loss'].append(l.item())

            ii = len(losses['perceptual_loss'])
            if ii % 5 == 0 and save_losses:
                print(f"Iteration: {ii}, Loss: {l.item()}")
            if ii % 10 == 0:
                vis_directions(direction, ii, out_dir=out_dir)

            optimizer.zero_grad()
            l.backward(retain_graph=True)
            return l
        
        optimizer.step(closure)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch}, Iterations: {len(losses['perceptual_loss'])}, LR: {current_lr}")

    if save_losses:
        print(f"Total iterations: {len(losses['perceptual_loss'])}")
        return direction, losses
    
    return direction


def diff_renorm(image):
    """Maps image back into [0,1]. Useful for visualising differences"""
    scale = 0.5/image.abs().max()
    image = image*scale
    image += 0.5
    return image


def vis_directions(im, iteration, out_dir='results'):
    # Ensure the directory exists
    im = diff_renorm(im)
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure()
    
    # Display the image
    if im.detach().cpu().numpy().ndim == 4:
        plt.imshow(im.detach().cpu().numpy()[0].transpose(1, 2, 0))
    else:
        plt.imshow(im.detach().cpu().numpy()[0])
        plt.colorbar()
    
    plt.axis('off')
    
    file_path = f"./{out_dir}/out_{iteration}.png"
    plt.savefig(file_path, bbox_inches='tight')
    
    plt.close()

def vis(im, text=False):
    plt.figure()
    if text:
        plt.title(text.split(', ')[0])
    if im.detach().cpu().numpy().ndim == 4:
        plt.imshow(im.detach().cpu().numpy()[0].transpose(1, 2, 0))
    else:
        plt.imshow(im.detach().cpu().numpy()[0])
        plt.colorbar()
    plt.axis('off')
    if text is not False:
        plt.savefig(
            text.split(', ')[0].replace(
                ' ', '').replace('/', ''), bbox_inches='tight')
