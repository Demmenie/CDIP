import numpy as np
from skimage.metrics import structural_similarity as ssim
import math
import scipy.fft
import scipy.io as sp

class FFT:

    """
    Does the Fourier Analysis used in experiments. This code was copied from
    MAGIC UNet, which is why it isn't commented further.

    Input:
        - normInData (np.array): Normalised input data.
    """

    def __init__(self, normInData):
        #### parameters used in Fourier Inversion (not needed for network) ####

        #permeability of free space
        self.mu0 = 1.25663706212e-6

        self.d = 14e-6 #vertical depth of current plane
        self.device_x_dimension=2e-3 #linear size of field of view (meters)
        self.hw_filter=True #turn Hann window function on/off

        self.noise=True #turn additive Gaussian noise on/off
        self.mean_noises=list(np.linspace(0,1,21)) #list of noise levels to test
        #levels represent std relative to max magnetic field pixel

        #**** 50 micron network ****

        self.z=50e-6 #average standoff distance
        #alternatively, you can use vector of exact values here
        self.kmax=90147.05882352941 #controls behavior of Hann function
        self.gauss_filt_strength=0.00012147058823529412 #Gaussian filter to reduce aliasing

        #****500 micron network****

        #z=500e-6
        #kmax=3029.1428571428573
        #gauss_filt_strength=0.00041142857142857143 #Gaussian filter to reduce aliasing


        # Set up Fourier space
        self.pixel_number_x = np.shape(normInData)[1]
        self.asize=np.shape(normInData)[0]
        self.dx = self.device_x_dimension/self.pixel_number_x
        self.dkx = 2*np.pi/(self.pixel_number_x*self.dx)
        self.k = np.arange(-self.dkx*self.pixel_number_x/2, self.dkx*self.pixel_number_x/2, self.dkx)
        self.k = np.where(np.abs(self.k)<1e-9,1e-9,self.k) #prevent weird behavior from k near 0
        self.kx, self.ky = np.meshgrid(self.k, self.k)

        self.standoff_distances=self.z*np.ones(self.asize)

    def dotheFFTsShift(self, data):
        numimages=np.shape(data)[0]
        bigger_out=[]
        out0=[]
        out1=[]
        out2=[]
        for i in range(numimages):
            temp=np.fft.fftshift(np.fft.fft2(data[i,:,:,0]))
            out0.append(temp)
        # for i in range(numimages):
        #     temp=np.fft.fftshift(np.fft.fft2(data[i,:,:,1]))
        #     out1.append(temp)
        # for i in range(numimages):
        #     temp=np.fft.fftshift(np.fft.fft2(data[i,:,:,2]))
        #     out2.append(temp)
        return np.asarray(out0)#, np.asarray(out1), np.asarray(out2)

    def dotheiFFTsShift(self, data):
        numimages=np.shape(data)[1]
        bigger_out=[]
        out0=[]
        out1=[]
        out2=[]
        for i in range(numimages):
            temp=np.fft.ifft2(np.fft.ifftshift(data[0][i,:,:]))
            out0.append(temp)
        # for i in range(numimages):
        #     temp=np.fft.ifft2(np.fft.ifftshift(data[1][i,:,:]))
        #     out1.append(temp)
        # for i in range(numimages):
        #     temp=np.fft.ifft2(np.fft.ifftshift(data[2][i,:,:]))
        #     out2.append(temp)
        return np.asarray(out0)# , np.asarray(out1), np.asarray(out2)

    def dotheiFFTsJShift(self, data):
        numimages=np.shape(data)[1]
        bigger_out=[]
        out0=[]
        out1=[]
        for i in range(numimages):
            temp=np.fft.ifft2(np.fft.ifftshift(data[0][i,:,:]))
            out0.append(temp)
        for i in range(numimages):
            temp=np.fft.ifft2(np.fft.ifftshift(data[1][i,:,:]))
            out1.append(temp)
        return np.asarray(out0), np.asarray(out1)

    def dotheFFTsJ(self, data):
        numimages=np.shape(data)[0]
        bigger_out=[]
        out0=[]
        out1=[]
        for i in range(numimages):
            temp=np.fft.fft2(data[i,:,:,0])
            out0.append(temp)
        for i in range(numimages):
            temp=np.fft.fft2(data[i,:,:,1])
            out1.append(temp)
        return np.asarray(out0), np.asarray(out1)

    def dotheFFTsJShift(self, data):
        numimages=np.shape(data)[0]
        bigger_out=[]
        out0=[]
        out1=[]
        for i in range(numimages):
            temp=np.fft.fftshift(np.fft.fft2(data[i,:,:,0]))
            out0.append(temp)
        for i in range(numimages):
            temp=np.fft.fftshift(np.fft.fft2(data[i,:,:,1]))
            out1.append(temp)
        return np.asarray(out0), np.asarray(out1)

    def blurBs(self, b, strength): #, bx, by, bz,
        kmagsquared = self.kx**2+self.ky**2
        blurfunction=np.exp(-0.25*kmagsquared*strength**2)
        return b*blurfunction # bx*blurfunction, by*blurfunction, bz*blurfunction

    def lessInfuriatingTensorProduct(self, twodarray, onedarray):
        temp=np.outer(onedarray, twodarray)
        out=[]
        for i in range(np.size(onedarray)):
            out.append(temp[i].reshape((np.shape(twodarray)[0], np.shape(twodarray)[1])))
        return np.asarray(out)

    def fourierb_from_fourierj(self, jx, jy):
        kmag = np.sqrt(self.kx**2+self.ky**2)
        exponentialargument=self.lessInfuriatingTensorProduct(kmag, self.standoff_distances)
        scalefactor = (self.mu0*self.d)/2
        print(np.shape(jy))
        bx = scalefactor*np.exp(-exponentialargument)*jy
        by = -scalefactor*np.exp(-exponentialargument)*jx
        jxjycurlcrossterm=(self.ky/kmag)*jx-(self.kx/kmag)*jy
        bz = -1j*scalefactor*np.exp(-exponentialargument)*(jxjycurlcrossterm)
        return bx, by, bz

    def fourierj_from_fourierb(self, b): # , bx, by, bz
        kmag = np.sqrt(self.kx**2+self.ky**2)
        exponentialargument=self.lessInfuriatingTensorProduct(kmag, self.standoff_distances)
        scalefactor = 2/(self.mu0*self.d)

        if self.hw_filter:
            hann2D=np.zeros([len(self.k),len(self.k)])
            for i in range(len(self.k)):
                for j in range(len(self.k)):

                    kmag_temp=np.sqrt(self.k[i]**2+self.k[j]**2);

                    if kmag_temp<self.kmax:
                        hann2D[i,j]=0.5*(1+math.cos((math.pi*kmag_temp/self.kmax)))
                    else:
                        hann2D[i,j]=1

            b=b*hann2D

        jx = -scalefactor*np.exp(exponentialargument)*b
        jy = scalefactor*np.exp(exponentialargument)*b
        return jx, jy

    def calculate_ssim_rmse_psnr(self, validationdata, fourierdata, predictiondata):
        ssim_vec_f_x=np.zeros([self.asize,1])
        ssim_vec_f_y=np.zeros_like(ssim_vec_f_x)
        ssim_vec_pred_x=np.zeros_like(ssim_vec_f_x)
        ssim_vec_pred_y=np.zeros_like(ssim_vec_f_x)
        rmse_vec_f_x=np.zeros_like(ssim_vec_f_x)
        rmse_vec_f_y=np.zeros_like(ssim_vec_f_x)
        rmse_vec_pred_x=np.zeros_like(ssim_vec_f_x)
        rmse_vec_pred_y=np.zeros_like(ssim_vec_f_x)
        psnr_vec_f_x=np.zeros_like(ssim_vec_f_x)
        psnr_vec_f_y=np.zeros_like(ssim_vec_f_x)
        psnr_vec_pred_x=np.zeros_like(ssim_vec_f_x)
        psnr_vec_pred_y=np.zeros_like(ssim_vec_f_x)

        for i in range(self.asize):
            vmax=np.max(abs(validationdata[i,:]))
            if vmax==0:
                vmax=1e7
            vmin=-vmax
            ssim_vec_f_x[i,0]=ssim(validationdata[i,:,:,0],fourierdata[i,:,:,0],data_range=2*vmax, multichannel=False)
            ssim_vec_f_y[i,0]=ssim(validationdata[i,:,:,1],fourierdata[i,:,:,1],data_range=2*vmax, multichannel=False)
            ssim_vec_pred_x[i,0]=ssim(validationdata[i,:,:,0], predictiondata[i,:,:,0],data_range=2*vmax, multichannel=False) #predictiondata[i,:,:,0]
            ssim_vec_pred_y[i,0]=ssim(validationdata[i,:,:,1],predictiondata[i,:,:,1],data_range=2*vmax, multichannel=False)

            rmse_array_f=np.sqrt(((validationdata[i,:,:,:]-fourierdata[i,:,:,:])/vmax)**2)
            rmse_array_pred=np.sqrt(((validationdata[i,:,:,:]-predictiondata[i,:,:,:])/vmax)**2)
            rmse_vec_f_x[i]=np.mean(rmse_array_f[:,:,0])
            rmse_vec_f_y[i]=np.mean(rmse_array_f[:,:,1])
            rmse_vec_pred_x[i]=np.mean(rmse_array_pred[:,:,0])
            rmse_vec_pred_y[i]=np.mean(rmse_array_pred[:,:,1])

            psnr_vec_f_x[i,0]=psnr(validationdata[i,:,:,0],fourierdata[i,:,:,0],data_range=2*vmax)
            psnr_vec_f_y[i,0]=psnr(validationdata[i,:,:,1],fourierdata[i,:,:,1],data_range=2*vmax)
            psnr_vec_pred_x[i,0]=psnr(validationdata[i,:,:,0],predictiondata[i,:,:,0],data_range=2*vmax)
            psnr_vec_pred_y[i,0]=psnr(validationdata[i,:,:,1],predictiondata[i,:,:,1],data_range=2*vmax)

        ssim_val_f_x=np.mean(ssim_vec_f_x)
        ssim_val_f_y=np.mean(ssim_vec_f_y)
        ssim_val_pred_x=np.mean(ssim_vec_pred_x)
        ssim_val_pred_y=np.mean(ssim_vec_pred_y)
        ssim_val_f = (ssim_val_f_x + ssim_val_f_y)/2
        ssim_val_pred = (ssim_val_pred_x + ssim_val_pred_y)/2

        rmse_val_f_x=np.mean(rmse_vec_f_x)
        rmse_val_f_y=np.mean(rmse_vec_f_y)
        rmse_val_pred_x=np.mean(rmse_vec_pred_x)
        rmse_val_pred_y=np.mean(rmse_vec_pred_y)
        rmse_val_f=(rmse_val_f_x + rmse_val_f_y)/2
        rmse_val_pred=(rmse_val_pred_x + rmse_val_pred_y)/2

        psnr_val_f_x=np.mean(filter_inf(psnr_vec_f_x))
        psnr_val_f_y=np.mean(filter_inf(psnr_vec_f_y))
        psnr_val_pred_x=np.mean(filter_inf(psnr_vec_pred_x))
        psnr_val_pred_y=np.mean(filter_inf(psnr_vec_pred_y))
        psnr_val_f = (psnr_val_f_x + psnr_val_f_y)/2
        psnr_val_pred = (psnr_val_pred_x + psnr_val_pred_y)/2

        return [ssim_val_f_x, ssim_val_f_y, ssim_val_pred_x, ssim_val_pred_y, ssim_val_f, ssim_val_pred], \
            [rmse_val_f_x, rmse_val_f_y, rmse_val_pred_x, rmse_val_pred_y, rmse_val_f, rmse_val_pred], \
                [psnr_val_f_x, psnr_val_f_y, psnr_val_pred_x, psnr_val_pred_y, psnr_val_f, psnr_val_pred]

    def filter_inf(self, mat):
        mat=np.asarray(mat)
        return mat[np.isfinite(mat)]