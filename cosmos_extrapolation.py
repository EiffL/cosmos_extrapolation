import os
import galsim
import numpy as np
from scipy.stats import kurtosis, norm, logistic, gaussian_kde
from scipy.special import gamma
import scipy.optimize as op
import pylab as plt

class sersic_model:
    """

    Parameters
    ----------

    cosmos_path: string
        Path to the cosmos training sample
    """
    def __init__(self, sample="25.2"):
        self.cosmos = galsim.COSMOSCatalog(sample=sample, exclusion_level="marginal")
        self.cat = self.cosmos.param_cat[self.cosmos.orig_index]

        # Compute mask to exclude some outliers
        self.mask = ((self.cat['sersicfit'][:,1] > 0) *
                (np.log10(self.cat['sersicfit'][:,0]) > -5) * (np.log10(self.cat['sersicfit'][:,0]) < 0))

    def fit(self, mag_range=[22., 23.5]):
        """
        Fits the model in the given magnitude_range

        Parameters
        ----------

        mag_range: list of float
            Range of mag_auto magnitudes to include in the fit
        """
        self.mag_range = mag_range

        # Index of galaxies in the magnitude range
        inds = self.mask * ((self.cat['mag_auto'] > mag_range[0]) *
                            (self.cat['mag_auto'] < mag_range[1]))

        # Extracts variables of interest
        mag = self.cat['mag_auto'][inds]
        I = np.log10(self.cat['sersicfit'][inds,0])
        R = np.log10(self.cat['sersicfit'][inds,1])
        n = np.log10(self.cat['sersicfit'][inds,2])
        q = np.log10(self.cat['sersicfit'][inds,3])

        def lnlike(theta, mag, x):
            """
            Log likelihood of the Logistic model with
            mean and standard deviation as affine functions of magnitude
            """
            a1, b1, a2,b2 = theta
            mu = a1 + b1*mag
            s = np.abs(a2 + b2*mag)
            return - np.sum( (x - mu)/s + np.log(s) + 2.*np.log(1 + np.exp(-(x -mu)/s)))

        nll = lambda *args: -lnlike(*args)
        theta0=(np.mean(I),0, 0.1, 0.1)
        bnds = ((None, None), (None, None), (None, None), (0, None))
        resfit = op.minimize(nll, theta0, args=(mag, I), bounds=bnds)
        print "Fit of intensities successfull: ",resfit['success']

        # Saving the parameters of the fit
        self.theta_I = resfit['x']

        # Fitting sersic profiles and axis ratios using a kde
        kde = gaussian_kde(np.array([n, q]))
        self.shape_kde = kde

        # Compute log10 of the total flux from the sersic
        mag_sersic = self._mag_sersic(I, R, n, q)

        # Fitting the residuals between mag_auto and the magnitude computed from
        # sersic fit using an affine function
        def mag_res(theta, mag, mag_s):
            """
            Computes residuals between mag_auto and mag_sersic magnitudes
            """
            a,b = theta
            return mag_s - (a +b*mag)

        resfit = op.least_squares(mag_res, (np.mean(mag - mag_sersic), 1), args=(mag, mag_sersic))
        print "Fit of sersic magnitudes successfull: ",resfit['success']
        self.theta_mag = resfit['x']

    def sample(self, mag):
        """
        Sample from the model given the specified magnitudes
        """
        ngal = len(mag)

        # Sample from the KDE, independently of the magnitude
        smp = self.shape_kde.resample(ngal)
        n = np.clip(smp[0,:],-1, 0.778)
        q = np.clip(smp[1,:],-1.2, 0)

        # Sample from the intensity distribution
        X = np.random.rand(ngal)
        s = self._I_s(mag)
        mu = self._I_mu(mag)
        I = s*( np.log(X) - np.log(1 - X)) + mu

        # Compute R by substration
        a,b = self.theta_mag
        en = 10**n
        k =  en * np.exp(0.6950 - 0.1789/en)
        R = -0.5*( (a + b*mag)/2.5 + ( n + np.log10(gamma(2*en) * np.exp(k) * k**(-2*en)) +
                            q + I + np.log10(2. * np.pi)))
        return 10**I, 10**R, 10**n, 10**q

    def _mag_sersic(self, I, R, n, q):
        """
        Return the magnitude computed from the sersic profile, up to a constant
        """
        en = 10**n
        k =  en * np.exp(0.6950 - 0.1789/en)
        mag_sersic = -2.5*( n + np.log10(gamma(2*en) * np.exp(k) * k**(-2*en)) +
                            q + 2*R + I + np.log10(2. * np.pi))
        return mag_sersic

    def _I_mu(self, mag):
        """
        Return the value of the mean log10 intensity as a function of magnitude
        """
        a1, b1, a2,b2 = self.theta_I
        return  a1 + b1*mag

    def _I_s(self, mag):
        """
        Return the scale of the Logistic function
        """
        a1, b1, a2,b2 = self.theta_I
        s = np.abs(a2 + b2*mag)
        return s

    def _I_std(self, mag):
        """
        Return the value of the standard deviation of log10 intensity as a function of magnitude
        """
        return  self._I_s(mag)/(np.sqrt(3.)/np.pi)

    def plot_fit_I(self, nbins=20):
        """
        Plots a comparison of the results of the fit to the input distribution
        """
        me = np.zeros(nbins)
        st = np.zeros(nbins)
        m =  np.zeros(nbins)
        k = np.zeros(nbins)

        # Extracts variables of interest
        mag = self.cat['mag_auto'][self.mask]
        I = np.log10(self.cat['sersicfit'][self.mask,0])
        R = np.log10(self.cat['sersicfit'][self.mask,1])
        n = np.log10(self.cat['sersicfit'][self.mask,2])
        q = np.log10(self.cat['sersicfit'][self.mask,3])

        plt.figure(figsize=(20,5))
        plt.subplot(141)
        for i in range(nbins):
            m_min = np.percentile(mag,i*5)
            m_max = np.percentile(mag,(i+1)*5)
            ind = (mag > m_min ) * (mag < m_max)
            m[i] = 0.5*(m_min +m_max )
            me[i] = np.mean(I[ind])
            st[i] = np.std(I[ind])
            k[i] = kurtosis(I[ind])
            plt.hist((I[ind]-me[i])/st[i],30,range=[-5,5],alpha=0.2, normed=True)
        y = np.linspace(-5,5)

        plt.plot(y, norm.pdf(y),'b', label='Gaussian')
        plt.plot(y, logistic.pdf(y,scale= np.sqrt(3)/np.pi),'r', label='Logistic')
        plt.legend()
        plt.title('Standardized $\log_{10}(I)$ in magnitude bins')

        plt.subplot(142)
        plt.plot(m, me,'+-')
        plt.plot(m,self._I_mu(m),'r--')
        plt.title('Mean $\log_{10}(I)$ ')
        plt.axvspan(self.mag_range[0], self.mag_range[1], alpha=0.2, color='k')

        plt.subplot(143)
        plt.plot(m, st,'+-')
        plt.plot(m,self._I_std(m),'r--')
        plt.axvspan(self.mag_range[0], self.mag_range[1], alpha=0.2, color='k')
        plt.title('Standard deviation of $\log_{10}(I)$ ')

        plt.subplot(144)
        plt.plot(m, k,'+-')
        plt.axhline(1.2,color='r')
        plt.axvspan(self.mag_range[0], self.mag_range[1], alpha=0.2, color='k')
        plt.title('Kurtosis')

    def plot_fit_kde(self):
        """
        Compares the kde estimation of the joint distribution of axis ratios
        and sersic indices
        """

        # Extracts variables of interest
        mag = self.cat['mag_auto'][self.mask]
        I = np.log10(self.cat['sersicfit'][self.mask,0])
        R = np.log10(self.cat['sersicfit'][self.mask,1])
        n = np.log10(self.cat['sersicfit'][self.mask,2])
        q = np.log10(self.cat['sersicfit'][self.mask,3])

        plt.figure(figsize=(15,5))
        plt.subplot(131)
        plt.hexbin(n,q,gridsize=50,cmap='jet',bins='log', extent=(-1,0.8,-1.2,0))
        plt.xlabel("$\log_{10}(n)$")
        plt.ylabel("$\log_{10}(q)$")
        plt.title('without magnitude cuts')
        plt.subplot(132)
        nobjs = len(n)
        s = self.shape_kde.resample(nobjs)
        plt.hexbin(np.clip(s[0,:],-1, 0.778) ,np.clip(s[1,:],-1.2,0),gridsize=50,cmap='jet',bins='log', extent=(-1,0.8,-1.2,0))
        plt.xlabel("$\log_{10}(n)$")
        plt.ylabel("$\log_{10}(q)$")
        plt.title('Sampled from KDE')
        plt.subplot(133)
        ind = (mag > self.mag_range[0] ) * (mag < self.mag_range[1])
        plt.hexbin(n[ind],q[ind],gridsize=50,cmap='jet',bins='log', extent=(-1,0.8,-1.2,0))
        plt.xlabel("$\log_{10}(n)$")
        plt.ylabel("$\log_{10}(q)$")
        plt.title('within fitting magnitude range')

        plt.figure(figsize=(15,5))
        plt.subplot(121)
        plt.hist(q[ind],50,label='within fitting region',normed=True,alpha=0.5)
        plt.hist(np.clip(s[1,:],-1.2,0),50,label='KDE samples',normed=True,alpha=0.5)
        plt.hist(q,50,label='without cuts',normed=True,alpha=0.5)
        plt.legend(loc=2)
        plt.xlabel("$\log_{10}(q)$")
        plt.subplot(122)
        plt.hist(n[ind],50,label='within fitting region',normed=True,alpha=0.5)
        plt.hist(np.clip(s[0,:],-1,0.778),50,label='KDE samples',normed=True,alpha=0.5)
        plt.hist(n,50,label='without cuts',normed=True,alpha=0.5)
        plt.legend(loc=2)
        plt.xlabel("$\log_{10}(n)$")

    def plot_fit_mag(self):
        """
        Plots the fit of sersic magnitudes to mag_auto ones
        """
        # Extracts variables of interest
        mag = self.cat['mag_auto'][self.mask]
        I = np.log10(self.cat['sersicfit'][self.mask,0])
        R = np.log10(self.cat['sersicfit'][self.mask,1])
        n = np.log10(self.cat['sersicfit'][self.mask,2])
        q = np.log10(self.cat['sersicfit'][self.mask,3])

        plt.figure()
        a, b = self.theta_mag
        plt.hexbin(mag,(a + b *mag) - self._mag_sersic(I, R, n, q), bins='log', cmap='Blues', extent=(17,25.2,-2,2))
        plt.colorbar(label='Log(N)')
        plt.axhline(0,color='red')
        plt.xlabel('mag_auto')
        plt.ylabel('mag_auto - mag_sersic, after affine correction')
