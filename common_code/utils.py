# Utilities to make running code simpler
from PIL import Image
from torchvision import transforms


def open_and_resize(image_name):
    im = Image.open(image_name).convert("RGB")
    im = im.resize(size=(224, 224))
    return im


def normalise_and_preprocess(im):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize
    ])
    return preprocess(im)


def open_and_preprocess(image_name):
    im = open_and_resize(image_name)
    # return normalise_and_preprocess(im)
    # add batch dim if needed
    im = normalise_and_preprocess(im)
    if len(im.shape) == 3:
        im = im.unsqueeze(0)
    return im
        

def open_and_preprocess_dino(image_name, img_size, patch_size):
    im = open_and_resize(image_name)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        normalize
    ])
    im = preprocess(im)

    # make the image divisible by the patch size
    w, h = im.shape[1] - im.shape[1] % patch_size, im.shape[2] - im.shape[2] % patch_size
    im = im[:, :w, :h].unsqueeze(0)

    return im

def preprocess_dino(im, img_size, patch_size):
    # im: [batch, channel, height, width]
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
    ])
    im = preprocess(im)
    w, h = im.shape[-2] - im.shape[-2] % patch_size, im.shape[-1] - im.shape[-1] % patch_size
    im = im[..., :w, :h]
    
    return im
