import numpy as np
import matplotlib.pyplot as plt

class RandomMask(object):
    """
    Class that holds and applies a completely random k-space mask on images.
    """

    def __init__(self, acceleration, seed=None):
        """
        acceleration = acceleration factor for the masking, if the acceleration is 2, then half the k-space points will not be masked, if it is 5 then only 20 % of the points will not be masked.
        seed = rng seed for reproducibility.
        """
        self.acceleration = acceleration
        self.rng = np.random.RandomState()
        if seed:
            self.rng.seed(seed)

    def __call__(self, kspace, newmask=True):
        """
        kspace = k-space distribution of points that needs to be masked. Can be 2D or 3D.
        """
        mask = self.get_mask(kspace)
        return kspace * mask.astype(np.float) + 0.0

    def get_mask(self, kspace):
        """
        kspace = k-space distribution of points that needs to be masked. Can be 2D or 3D.
        """
        return self.rng.uniform(size=kspace.shape) < (1/self.acceleration)

class CenteredRandomMask(RandomMask):
    """
    Same as RandomMask but ensures center of kspace is fully sampled
    """
    def __init__(self,acceleration, center_fraction, seed=None):
        """
        docstring
        """
        self.acceleration = acceleration
        self.rng = np.random.RandomState()
        if seed:
            self.rng.seed(seed)
        self.center_fraction = center_fraction

    def get_mask(self, kspace):
        """
        kspace = k-space distribution of points that needs to be masked. Can be 2D.
        expected to have regular shape (shape[0]==shape[1])
        """
        #TODO code this better, generalise to ND
        size = kspace.shape[0]*kspace.shape[1]
        num_low_freqs = int(round(kspace.shape[0]*self.center_fraction))
        prob = (size/(size-(num_low_freqs**2)))/self.acceleration

        mask = self.rng.uniform(size=kspace.shape) < prob
        low = (kspace.shape[0] - num_low_freqs)/2
        high = (kspace.shape[0] + num_low_freqs)/2
        for i in range(kspace.shape[0]):
            for j in range(kspace.shape[1]):
                if i >= low and i<=high and j>=low and j<= high:
                    mask[i,j] = True
        return mask


class PolynomialMaskGenerator(RandomMask):
    """
    docstring
    """
    
    def __init__(self, imsize, poly=8,sampling_factor=0.15,distType=2,radius=0.02,dim=1):
        """
        docstring
        """
        self.imsize = imsize
        if dim==1:
            size = imsize[0]
        else:
            size = imsize
        self.dim = dim
        self.sampling_factor = sampling_factor
        self.pdf, self.pdfval =  self.genPDF(size,poly=poly,sampling_factor=sampling_factor,distType=distType,radius=radius,dim=dim)

    def get_mask(self,kspace,niter=1000,deviation=0.05):
        if self.dim==2:
            dev = np.product(self.imsize)*deviation
        else:
            dev = self.imsize[0]*deviation
        minIntrVec, stat, actualundersampling = self.genSampling(self.pdf,niter,dev)
        if self.dim==1:
            tmp = np.ones(self.imsize)
            for i in range(tmp.shape[0]):
                tmp[i,:]*=minIntrVec
            minIntrVec = tmp
        return minIntrVec

    def genPDF(self, imsize,poly,sampling_factor,distType,radius,display=False,dim=1):
        """
        generates a pdf for 1d or 2d random sampling pattern with polynomial variable density sampling
        from Lustig's implementation on MATLAB
        imsize : size of the image
        poly : power of the polynomial
        sampling_factor : partial sampling factor e.g. 0.5 for half
        distType : 1 or 2 for L1 or L2 distance measure
        radius : radius of fully sampled center
        display : display output

        outputs : pdf, val
        pdf : the pdf
        val : min samping density
        """
        minval = 0.
        maxval= 1.
        val = 0.5
        if dim==2:
            sx = imsize[0]
            sy = imsize[1]
            sampling_factor = np.floor(sampling_factor*sx*sy)
            x,y = np.meshgrid(np.linspace(-1,1,sx),np.linspace(-1,1,sy))
            if distType==1:
                r = max(abs(x),abs(y))
            elif distType==2:
                r = np.sqrt(np.square(x)+np.square(y))
                r = r/np.amax(np.abs(r[:]))
        else:
            sampling_factor = np.floor(sampling_factor*imsize)
            r = np.abs(np.linspace(-1,1,imsize))

        idx = np.where(r<radius)

        pdf = np.power((1.-r),poly)
        pdf[idx] = 1
        if np.floor(np.sum(pdf)) > sampling_factor:
            raise ValueError('infeasible without undersampling, increase poly')

        #begin bisection
        while True:
            val = minval/2 + maxval/2
            pdf = np.power((1-r),poly) + val
            pdf[np.where(pdf>1.)] = 1.
            pdf[idx] = 1.
            N = np.floor(np.sum(pdf[:]))
            if N > sampling_factor:
                maxval = val
            elif N < sampling_factor:
                minval=val
            elif N==sampling_factor:
                break

        if display:
            fig, axs = plt.subplots(1,2, figsize=(20, 20))
            fig.subplots_adjust(hspace=0.1, wspace=0.1)
            axs = axs.ravel()

            if dim==2:
                axs[0].imshow(pdf,cmap='gray')
                axs[1].plot(pdf[int(pdf.shape[0]/2+1),:])
            else:
                axs[0].plot(pdf)
            plt.show()

        return pdf, val



    def genSampling(self, pdf,niter,deviation):
        pdf[np.where(pdf>1.)] = 1.
        K = np.sum(pdf)

        minIntr = 1e99
        minIntrVec = np.zeros(pdf.shape)
        stat = np.zeros(niter)
        for i in range(niter):
            tmp = np.zeros(pdf.shape)
            while np.abs(np.sum(tmp) - K) > deviation:
                tmp = np.random.rand(*pdf.shape)<pdf
            if np.ndim(tmp)==2:
                TMP = np.fft.ifft2(np.divide(tmp,pdf))
            else:
                TMP = np.fft.ifft(np.divide(tmp,pdf))
            if np.max(np.abs(TMP[1:])) < minIntr:
                minIntr = np.max(np.abs(TMP[1:]))
                minIntrVec = tmp
            stat[i] = np.max(np.abs(TMP[1:]))

        actualundersampling = np.sum(minIntrVec)/np.product(minIntrVec.shape)

        return minIntrVec, stat, actualundersampling

