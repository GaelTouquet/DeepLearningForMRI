import numpy as np

def ifft(kdata):
    """
    Simplifies the use of ifft.
    """
    tmp = np.fft.ifftshift(kdata)
    tmp = np.fft.ifft2(tmp)
    return np.fft.fftshift(tmp)

def crop(img):
    """
    Simplifies the use of cropping to fastMRI image size.
    """
    xstart = int((img.shape[0]-320)/2)
    xend = int(img.shape[0]-xstart)
    ystart = int((img.shape[1]-320)/2)
    yend = int(img.shape[1]-ystart)
    return img[xstart:xend,ystart:yend]

def rss(imgs):
    """
    Computes and return RSS.
    """
    return np.sqrt(np.sum(np.square(abs(imgs)), axis=0))

def image_from_kspace(kdata,multicoil=True):
    if multicoil:
        return rss(crop(ifft(kdata)))
    else:
        return crop(ifft(kdata))