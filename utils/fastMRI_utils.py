import numpy as np


def fft(image):
    tmp = np.fft.fftshift(image)
    tmp = np.fft.fft2(tmp)
    return np.fft.ifftshift(tmp)

def ifft(kdata):
    """
    Simplifies the use of ifft.
    """
    tmp = np.fft.ifftshift(kdata)
    tmp = np.fft.ifft2(tmp)
    return np.fft.fftshift(tmp)


def crop(img, size=(320,320)):
    """
    Simplifies the use of cropping to the given image size.
    """
    xstart = int((img.shape[0]-size[0])/2)
    xend = int(img.shape[0]-xstart)
    ystart = int((img.shape[1]-size[1])/2)
    yend = int(img.shape[1]-ystart)
    return img[xstart:xend, ystart:yend]

def crop_from_kspace(kspace, size=(320,320)):
    image = ifft(kspace)
    cropped_image = crop(image,size=size)
    return fft(cropped_image)

def rss(imgs):
    """
    Computes and return RSS.
    """
    return np.sqrt(np.sum(np.square(abs(imgs)), axis=0))


def image_from_kspace(kdata, multicoil=True, mask=None, normalise=False,image_shape=(320,320), output_kspace=False):
    """
    Provides the image from the given kdata. Also applies the mask if given one.
    """
    if multicoil:
        if mask:
             the_mask = mask.get_mask(kdata[0])
        images = []
        for coil in kdata:
            if mask:
                image = the_mask * coil + 0.0
            else:
                image = coil
            images.append(crop(ifft(image),size=image_shape))
        image = np.array(images)
    else:
        if mask:
            data = mask(kdata)
        else:
            data = kdata
        image = crop(ifft(data),size=image_shape)

    if normalise:
        image *= 1./np.amax(image)

    if output_kspace:
        image = fft(image)

    return image
