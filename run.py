# Loading libraries

# Image libraries and image processing functions
import torchvision
from scipy.ndimage import gaussian_filter
import numpy as np

# Our method code
from common_code.mul_perceptualFunc_Final import *
from common_code import utils

# Plotting libraries
import matplotlib.pyplot as plt

# Simple function wrapper
from functools import partial

def perceptual_dino(args, model=None,layerLists=['16', '19', '22', '25', '29', '32'], img_size=(224, 224), patch_size=14):
    '''
    perceptual function that returns a saliency map. 
    k = iterations
    layerLists = the layers to regularise with the perceptual loss
    ga = weight of perceptual loss 
    '''
    
    # If the model is not specified assume vgg19
    if model==None:
        model = torchvision.models.vgg19_bn(pretrained=True)
        model.requires_grad=False
        model.eval()
    
    # load the image
    img_variable = utils.open_and_preprocess(args.image)
    img_dino = utils.open_and_preprocess_dino(args.image, img_size, patch_size)
    preprocess_dino = partial(utils.preprocess_dino, img_size=img_size, patch_size=patch_size)

    # Load onto GPU
    img_variable=img_variable.cuda()
    img_dino=img_dino.cuda()
    model=model.cuda()

    # create the perceptual loss with the required parameters
    # loss=create_perceptual_loss2(-2,img_variable,model,gamma=ga,scalar=sa,layers=layerLists)
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    for p in dino_model.parameters():
        p.requires_grad = False
    dino_model.cuda()
    dino_model.eval()
    loss = create_perceptual_loss_dino(-2, preprocess_dino, img_variable, img_dino, model, dino_model, gamma=args.ga, scalar=args.sa, dino_scalar=args.ds, target_scalar=args.ts, nat_scale=args.ns, layers=layerLists, targetID=args.targetID)  

    # optimise the loss to find the adv. perturbation
    c,losses = dino_find_direction(loss,img_variable,iterations=args.k, epochs=args.epochs,save_losses=True, out_dir=args.out_dir)
 
    # Take pixelwise euclidean distance to get the saliency map
    res=torch.sqrt(((c - img_variable)**2).mean(1))
    res=res.squeeze().cpu().detach().numpy()
    return c, res, losses

def perceptual(image_name,k=100,model=None,layerLists=['16', '19', '22', '25', '29', '32'],sa=1,ga=10000):
    '''
    perceptual function that returns a saliency map. 
    k = iterations
    layerLists = the layers to regularise with the perceptual loss
    ga = weight of perceptual loss 
    '''
    
    # If the model is not specified assume vgg19
    if model==None:
        model = torchvision.models.vgg19_bn(pretrained=True)
        model.requires_grad=False
        model.eval()
    
    # load the image
    img_variable = utils.open_and_preprocess(image_name)

    # Load onto GPU
    img_variable=img_variable.cuda()
    model=model.cuda()
    # create the perceptual loss with the required parameters
    loss=create_perceptual_loss2(-2,img_variable,model,gamma=ga,scalar=sa,layers=layerLists)

    # optimise the loss to find the adv. perturbation
    c=find_direction(loss,img_variable,iterations=k)
 
    # Take pixelwise euclidean distance to get the saliency map
    res=torch.sqrt(((c - img_variable)**2).mean(1))
    res=res.squeeze().cpu().detach().numpy()
    return c, res


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Perceptual Ball')
    parser.add_argument('--image', type=str, default='ILSVRC2012_val_00000051.JPEG', help='Image name')
    parser.add_argument('--out_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--k', type=int, default=50, help='Number of iterations')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--ga', type=float, default=10000, help='Weight of perceptual loss')
    parser.add_argument('--sa', type=float, default=1, help='Weight of L2 loss')
    parser.add_argument('--ds', type=float, default=100, help='Dino scalar')
    parser.add_argument('--targetID', type=int, default=None, help='Target ID')
    parser.add_argument('--ts', type=float, default=100, help='Target scalar')
    parser.add_argument('--ns', type=float, default=100, help='Natural scale')
    return parser.parse_args()

def _plot_losses(losses, img_path="losses.pdf"):
    # Losses is a dictionary with keys 'perceptual', 'prior', 'dino', 'brisque', 'label'
    # Plot losses on the same graph
    plt.plot(losses['perceptual_loss'], label='Perceptual')
    plt.plot(losses['prior_loss'], label='Prior')
    plt.plot(losses['dino_loss'], label='Dino')
    plt.plot(losses['brisque_loss'], label='Brisque')
    plt.plot(losses['label_loss'], label='Label')
    plt.plot(losses['total_loss'], label='Total')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(img_path)

def main():
    # Initializations
    args = parse_args()

    print(f"Arguments: {args}")
    os.makedirs(args.out_dir, exist_ok=True)
    # Model
    premodel = torchvision.models.vgg19_bn(pretrained=True)
    premodel.requires_grad=False
    premodel.eval()

    # Layers and Image
    layerAll=['2','5','9','12','16','19','22','25','29','32','35','38','42','45','48','51']
    # image name
    im = utils.open_and_resize(args.image)

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im)
    # Save image
    fig.savefig(f'./{args.out_dir}/original.png', dpi=100)

    # Run
    # layers to regularise
    layerslist=["0-1-2-3-4-5-6-7-8-9-10-11-12"]

    # get layers
    # indexes layerAll
    layers=[layerAll[int(k)] for k in layerslist[0].split('-')]

    # run adversarial perturbation on the perceptual ball
    adv, res, losses = perceptual_dino(args,
                               model=premodel,
                               layerLists=layers,
                               )

    # gaussian blur on image
    mat1 = gaussian_filter(res, sigma=2)

    # un-normalise
    aim = adv[0].cpu().detach().numpy().transpose(1,2,0)
    aim = (aim - aim.min())/(aim.max()-aim.min())
    aim = (aim*255).astype(np.uint8)

    adv_response = premodel(adv.cuda())
    image = utils.open_and_preprocess(args.image)
    response = premodel(image.cuda())

    print(f"Original response on target: {response[0, args.targetID].item()}")
    print(f"Adversarial response on target: {adv_response[0, args.targetID].item()}")

    # Plot losses and save figure as pdf
    # Clear
    plt.clf()
    _plot_losses(losses, f"./{args.out_dir}/losses.pdf")

    # Plot saliency map and adversarial perturbation
    plt.clf()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(mat1.squeeze())
    ax[0].set_title('Saliency map')
    ax[0].axis('off')
    ax[1].imshow(aim)
    ax[1].set_title('Adversarial perturbation')
    ax[1].axis('off')
    # Save image
    fig.savefig(f'./{args.out_dir}/saliency_map_adversarial_perturbation.png', dpi=100)

if __name__ == '__main__':
    main()