# additional functions and helpful libraries for SURP2023

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
    
    return (np.array([x,y])),subimage)

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