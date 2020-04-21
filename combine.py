import numpy as np
from astropy.io import fits
#import montage_wrapper as montage
import aplpy
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.visualization import MinMaxInterval
from astropy.visualization import PercentileInterval
from astropy.visualization import AsymmetricPercentileInterval
from astropy.visualization import ZScaleInterval
from astropy.visualization import SqrtStretch
from astropy.visualization import LogStretch
from astropy.visualization import AsinhStretch

# This is a very manual python2 script.
# Two-step process:
# First, comment-out Section 1 below, run it with suitable headers
# Then, Section 2 to combine the images produced.
# Montage python wrappers are required.
# The code should be refactored in two methods, etc. when time allows.


#'''
#r = fits.open('mosaic_i.fits')[0]
#g = fits.open('mosaic_r.fits')[0]
#b = fits.open('mosaic_g.fits')[0]

#wsrt = fits.open('/Users/shulevski/Desktop/A1318_WENSS.fits')
#nvss = fits.open('/Users/shulevski/Desktop/A1318_NVSS.fits')

#radio = fits.open('LOFAR_150MHz_crop.fits')[0]
'''
# Section 1

montage.commands.mGetHdr('mosaic_r.fits', 'r_header')
#montage.commands.mGetHdr('A1318_PNS_r.fits', 'r_header')

#montage.wrappers.reproject('/home/shulevski/Desktop/P173+55-mosaic.fits', 'rd_reprojected.fits' , header='r_header') # A1318
montage.wrappers.reproject('/home/shulevski/Documents/Research/Projects/3C236/3C236_DDFacet_Nov2018/Image_SC3_iter3/3C236_man_img_SC3-MFS-image-pb.fits', 'rd_reprojected.fits' , header='r_header')
montage.wrappers.reproject('mosaic_i.fits', 'r_reprojected.fits' , header='r_header')
montage.wrappers.reproject('mosaic_r.fits', 'g_reprojected.fits' , header='r_header') 
# different SDSS bands are shifted wrt to one another, reproject using the same header for all
montage.wrappers.reproject('mosaic_g.fits', 'b_reprojected.fits' , header='r_header')

#montage.wrappers.reproject('A1318_PNS_i.fits', 'r_reprojected.fits' , header='r_header')
#montage.wrappers.reproject('A1318_PNS_r.fits', 'g_reprojected.fits' , header='r_header')
#montage.wrappers.reproject('A1318_PNS_g.fits', 'b_reprojected.fits' , header='r_header')

#montage.wrappers.reproject('/Users/shulevski/Desktop/A1318_WENSS.fits', 'A1318_WENSS_reprojected.fits' , header='r_header')
#montage.wrappers.reproject('/Users/shulevski/Desktop/A1318_NVSS.fits', 'A1318_NVSS_reprojected.fits' , header='r_header')

#A1318
#montage.mSubimage('rd_reprojected.fits', 'rd_reprojected_crop.fits', 173.98152, 55.09408, 0.48)
#montage.mSubimage('r_reprojected.fits', 'r_reprojected_crop.fits', 173.98152, 55.09408, 0.48)
#montage.mSubimage('g_reprojected.fits', 'g_reprojected_crop.fits', 173.98152, 55.09408, 0.48)
#montage.mSubimage('b_reprojected.fits', 'b_reprojected_crop.fits', 173.98152, 55.09408, 0.48)

#3C236
montage.mSubimage('rd_reprojected.fits', 'rd_reprojected_crop.fits', 151.56542, 34.866214, 0.66)
montage.mSubimage('r_reprojected.fits', 'r_reprojected_crop.fits', 151.56542, 34.866214, 0.66)
montage.mSubimage('g_reprojected.fits', 'g_reprojected_crop.fits', 151.56542, 34.866214, 0.66)
montage.mSubimage('b_reprojected.fits', 'b_reprojected_crop.fits', 151.56542, 34.866214, 0.66)

#montage.mSubimage('A1318_WENSS_reprojected.fits', 'A1318_WENSS_reprojected_crop.fits', 173.98152, 55.09408, 0.3)
#montage.mSubimage('A1318_NVSS_reprojected.fits', 'A1318_NVSS_reprojected_crop.fits', 173.98152, 55.09408, 0.3)
'''

# Section 2

rd_r = fits.open('rd_reprojected_crop.fits')[0]
r_r = fits.open('r_reprojected_crop.fits')[0]
g_r = fits.open('g_reprojected_crop.fits')[0]
b_r = fits.open('g_reprojected_crop.fits')[0]

#r_r = fits.open('A1318_PNS_i.fits')[0]
#g_r = fits.open('A1318_PNS_r.fits')[0]
#b_r = fits.open('A1318_PNS_g.fits')[0]

print "Blue comp: ", b_r.data.min(), b_r.data.max(), b_r.data.mean()


interval = MinMaxInterval()
pinterval = PercentileInterval(99.)
#rdapinterval = AsymmetricPercentileInterval(91., 99.98)
rdapinterval = AsymmetricPercentileInterval(92., 99.991)
#oapinterval = AsymmetricPercentileInterval(65, 99.89999999)
oapinterval = AsymmetricPercentileInterval(97.7, 99.998)
zsinterval = ZScaleInterval()
sstretch = SqrtStretch()
lstretch = LogStretch()
astretch = AsinhStretch()

rd_t = lstretch(rdapinterval(rd_r.data)) * 255.
r_t = lstretch(oapinterval(r_r.data)) * 255.
g_t = lstretch(oapinterval(g_r.data)) * 255.
b_t = lstretch(oapinterval(b_r.data)) * 255.

#red = interval(0.80 * rd_t + r_t) * 255.
#green = interval(0.1 * rd_t + g_t) * 255.
#blue = interval(0.1 * rd_t + b_t) * 255.

red = interval(r_t) * 255.
green = interval(g_t) * 255.
blue = interval(b_t) * 255.

#red = r_t
#green = g_t
#blue = b_t

print "Red: ", red.min(), red.max(), red.mean()
print "Green :", green.min(), green.max(), green.mean()
print "Blue: ", blue.min(), blue.max(), blue.mean()


fits.writeto('red.fits', red, rd_r.header)
fits.writeto('green.fits', green, rd_r.header)
fits.writeto('blue.fits', blue, rd_r.header)

aplpy.make_rgb_cube(['red.fits', 'green.fits', 'blue.fits'], 'color.fits')

aplpy.make_rgb_image('color.fits','color.png', embed_avm_tags=True) # play with limits

#aplpy.make_rgb_image('color.fits','color.png')
#'''
'''

#dd = fits.open('color_2d.fits')[0]
#print dd.header

#f = aplpy.FITSFigure('color.png',header=dd.header)

f = aplpy.FITSFigure('color_2d.fits')
#f.show_grayscale(invert=True, stretch='linear')
#f1 = aplpy.FITSFigure('A1318_WENSS_reprojected_crop.fits')
#f2 = aplpy.FITSFigure('A1318_NVSS_reprojected_crop.fits')


factor = 5.
factor1 = 3.
factor2 = 1.
l_sigma = 100.e-6

w_sigma = 18.e-3
n_sigma = 0.4e-3

levno = 10
levbase = 2.
levels = np.power(np.sqrt(levbase), range(levno))
levels = np.insert(levels, 0, -levels[0], axis=0)

f.show_contour('/Users/shulevski/Desktop/A1291_LOFAR.fits', levels=factor * l_sigma * levels, colors=['red'], linewidths=1.)
f.show_contour('/Users/shulevski/Desktop/A1318_WENSS.fits', levels=factor2 * w_sigma * levels, colors=['green'], linewidths=2.)
f.show_contour('/Users/shulevski/Desktop/A1318_NVSS.fits', levels=factor1 * n_sigma * levels, colors=['blue'], linewidths=2.)


cl_x = [174.0146, 173.58849, 174.03204, 174.26407, 173.96542]
cl_y = [55.07537, 55.0216, 55.23907, 55.12944, 55.09233]
rad = [0.193573364, 0.030456377, 0.038243168, 0.055894309, 0.075]


f.show_rgb('color.png')

f.show_circles(174.11, 54.96, 0.003, edgecolor='green', linewidth=2)
f.show_circles(174.00125, 55.03736, 0.003, edgecolor='magenta', linewidth=2)
#f.show_markers([174.11],[54.96], edgecolor='green', marker='X', facecolor='green', s=125)
#f.show_markers([174.00125],[55.03736], edgecolor='magenta', marker='X', facecolor='magenta', s=125)
#f.show_markers([173.96488],[55.09192], edgecolor='magenta', marker='X', facecolor='magenta', s=85)



f.show_circles(cl_x[1:4], cl_y[1:4], rad[1:4], edgecolor='white', linewidth=1, linestyle='dashed')
f.show_circles(cl_x[0], cl_y[0], rad[0], edgecolor='green', linewidth=1, linestyle='dashed')
f.show_circles(cl_x[4], cl_y[4], rad[4], edgecolor='yellow', linewidth=1, linestyle='dashed')
##f = aplpy.FITSFigure('color.png')

#f.show_rgb() # with PyAVM
#f.set_title('Abell 1318')
#f.set_theme('pretty')
plt.show()
'''
