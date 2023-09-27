# additional functions and helpful libraries for SURP2023

# useful libraries
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits # handeling FITS file
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
# photutils functions
from photutils.isophote import EllipseGeometry # used as initial guess for the ellipse
from photutils.aperture import EllipticalAperture # show initial ellipse guess
from photutils.isophote import Ellipse # fitting model
from photutils.isophote import build_ellipse_model # build ellipse model
# from photutils.isophote import isophote 
from photutils.isophote.isophote import Isophote # surface brightness library
from photutils.isophote import EllipseSample # used to create BV ellipses
from photutils.isophote import IsophoteList
from photutils.isophote.sample import CentralEllipseSample
from photutils.isophote.fitter import CentralEllipseFitter
import scipy.optimize as sco # fitting the sersic function

# make initial location of dwarf galaxies easier
def RADEC_to_source(RA,DEC,image_path,cutout):
    '''The purpose of this function is to make location of sources in a FITS file easier to locate.
    Input:
        RA/DEC: RA and DEC location in the FITS file in deg. dtype: str
        image_path: path of the image being searched. makes for less memory usage. dtype: str
    Output:
        subimage: sub-image of the source
        pix: pixel location of the source on the image. dtype: int'''
    # convert to degrees from hour/min/sec
    c = SkyCoord(RA+DEC, unit=(u.hourangle, u.deg))
    print('RA/DEC converted to deg:',c)
    # single out pixel location
    image = fits.open(image_path)
    image_dat = image[0].data
    image_head = image[0].header
    w = WCS(image_head)
    x, y = w.world_to_pixel(c)
    
    # compute sub-image (200x200 pixel cutouts to match the paper)
    subimage = image_dat[int(y)-cutout:int(y)+cutout,int(x)-cutout:int(x)+cutout]
    
    return (np.array([x,y]),subimage)

def pix_to_source(pix,image_path,cutout):
    '''The purpose of this function is to quickly check if a supposed target is a dwarf galaxy
    Input:
        pix: x,y position of the source. dtype: int
        image_path: path of the image being searched. makes for less memory usage. dtype: str
    Output:
        subimage: sub-image of the source
        sky: RA/DEC of the image. dtype: int'''
    # single out pixel location
    image = fits.open(image_path)
    image_dat = image[0].data
    image_head = image[0].header
    w = WCS(image_head)
    sky = w.pixel_to_world(pix[0],pix[1])
    
    # compute sub-image (200x200 pixel cutouts to match the paper)
    subimage = image_dat[int(pix[1])-cutout:int(pix[1])+cutout,int(pix[0])-cutout:int(pix[0])+cutout]
    
    return (sky,subimage)

# define a function to read in each of the results from tony's txt file
def file_reader(file_path,lines,img_path):
    '''input:
        file_path: a string of the file path to be opened
        lines: array with the number of lines to be read (not indexed at 0)
    returns:
        coords: coordinates of object in RA/DEC
        RADEC: pixel coordinates of the target'''
    coords = []
    RADEC = []
    for i in range(np.min(lines),np.max(lines)):
        f = open(file_path)
        k = f.readlines()[i][64]
        f.close()
        if (k == 'A') or (k == 'B'):
            f = open(file_path)
            a = f.readlines()[i][7:18]
            f.close()
            f = open(file_path)
            b = f.readlines()[i][19:31]
            f.close()
            
            coords.append(np.array([a,b]))
            RADEC.append(RADEC_to_source(a,b,img_path,30))
            
    print(len(coords))
    print(len(RADEC))
    return (np.array(coords),np.array(RADEC))

def fit_B(DATA,isolist_I,x0,y0,sma,eps,pa):
    '''Computes the isolist of the B band using prior information in I band'''
    isolist2= []
    for isoB in isolist_I[1:]:
        geo = isoB.sample.geometry
        sample = EllipseSample(DATA, geo.sma, geometry=geo, sclip=sclip, nclip=2, integrmode='median')
        sample.update(0)
        iso_B = Isophote(sample, 0, True, 0)
        isolist2.append(iso_B)
    isolist2 = IsophoteList(isolist2)
    g2 = EllipseGeometry(x0=x0, y0=y0, sma=sma, eps=eps, pa=pa * np.pi / 180.0)
    sample = CentralEllipseSample(DATA,0,geometry=g2)
    fitter = CentralEllipseFitter(sample)
    center = fitter.fit()
    isolist2.append(center)
    isolist2.sort()
    return isolist2

def fit_V(DATA,isolist_I,x0,y0,sma,eps,pa):
    '''Computes the isolist of the V band using prior information in I band'''
    isolist3 = []
    for isoV in isolist_I[1:]:
        geo = isoV.sample.geometry
        sample = EllipseSample(DATA, geo.sma, geometry=geo, sclip=sclip, nclip=2, integrmode='median')
        sample.update(0)
        iso_V = Isophote(sample, 0, True, 0)
        isolist3.append(iso_V)
    isolist3 = IsophoteList(isolist3)
    g3 = EllipseGeometry(x0=x0, y0=y0, sma=sma, eps=eps, pa=pa * np.pi / 180.0)
    sample = CentralEllipseSample(DATA,0,geometry=g3)
    fitter = CentralEllipseFitter(sample)
    center = fitter.fit()
    isolist3.append(center)
    isolist3.sort()
    return isolist3

def magnitude(flux):
    '''calculates the magnitude moving radially outward from the center of a galaxy'''
    mag = 2.5*np.log10(flux/0.16)
    return mag
def mag_err(flux_err):
    '''calculates the error of the magnitudes'''
    mag_err = 2.5*np.log10(flux_err/0.16)
    return mag_err

# Fitting Sersic function (Sersic function fits to I band)     **** rework func, r is x data points since its radius outwards
def sersic(r,mu0I,r0I,nI):
    '''Sersic functionto fit to the I band of the dwarf galaxy data
    input:
        muI: surface brightness (returned)
        mu0I: central surface brightness
        r: radius, use the data from ellipse fitting
        r0I: scale length
        nI: sersic index
        '''
    muI = (mu0I + 1.0857*(r/r0I)**(1/nI))
    return muI

# calculate chi-square and chi-square reduced
# defining chi square function
def chi_square(y_measured, y_expected,errors):
    return np.sum( np.power((y_measured - y_expected),2) / np.power(errors,2) )
# define chi square reduced
def chi_square_reduced(y_measured,y_expected,errors,number_parameters):
    return chi_square(y_measured,y_expected,errors)/(len(y_measured - number_parameters))

def eff_rad(n,r0):
    Bn = 0.868*n - 0.142
    eff_rad = r0*(2.3026*Bn)**n
    return eff_rad

