import aplpy
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
from astropy import units as u
import matplotlib.pyplot as plt

#'''
# 3C 236
#path = '/Users/shulevski/Documents/Research/Projects/3C236/LOFAR_Factor_images_Feb_2017/dataset/re-regridding/'
path = '/home/shulevski/Documents/Research/Projects/3C236/LOFAR_Factor_images_Feb_2018/'
path_hires = '/home/shulevski/Documents/Research/Projects/3C236/3C236_DDFacet_Nov2018/Image_SC3_iter3/'

lofar_im = '3C236_LOFAR.fits'
#inset_im = 'imtry1_9-MFS-image.fits'
#lofar_im = 'imtry1_9-MFS-image.fits'
#inset_im = '3C236_man_img_SC2-MFS-image.fits'
#lofar_im = '3C236_man_img_SC2-MFS-image.fits'
inset_im = '3C236_man_img_SC3-MFS-image-pb.fits'
lofar_im = '3C236_man_img_SC4_midres_default-MFS-image-pb.fits'

#lev_im = '3C236_LOFAR_AGES.fits'
#lev_im = 'LSC.fits'
#lev_im = '3C236_LOFAR_48asecorig_sm.fits'
#lofar_im = 'spix.fits'
#inset_im = 'spix.fits'

#lofar_im = 'curv_err.fits'
#inset_im = 'curv_err.fits'
#'''

fig = plt.figure(figsize=(11, 9))

'''
# Regions of interest

#lofar_im = 'LSC.fits'
#lofar_im = '3C236_LOFAR.fits'

im = aplpy.FITSFigure(path + lofar_im, figure=fig, dimensions=[0, 1], slices=[0, 0])
im.recenter(151.55, 34.90, radius=0.4)
im.show_grayscale(invert=True, stretch='linear', vmin=0.001, vmax=0.1)
im.axis_labels.set_font(size=18)
im.tick_labels.set_font(size=16)

factor = 20.
sigma = 3e-3
levno = 10
levbase = 2.
levels = np.power(np.sqrt(levbase), range(levno))	
im.show_contour(path+lofar_im, levels=factor * sigma * levels, colors=['black'], linewidths=1.1)

#im.show_rectangles(np.array([151.885108, 151.787620, 151.697969, 151.455176, 151.358325, 151.341375, 151.319955, 151.254662]), np.array([34.710602, 34.752080, 34.788003, 34.904841, 35.020029, 34.993528, 34.974135, 35.041590]), np.array([96.80, 79.56, 97.02, 92.67, 57.50, 56.80, 56.82, 93.13])/3600., np.array([76.52, 142.15, 94.78, 88.50, 79.49, 71.35, 75.56, 117.17])/3600., edgecolor='black', linewidth=2)

#im.show_lines([np.array([[151.651464, 151.727017, 151.771241, 151.801138, 151.829530, 151.867973, 151.929169], [34.814827, 34.772158, 34.768355, 34.749319, 34.713779, 34.700485, 34.678661]]), np.array([[151.438014, 151.354995, 151.293367, 151.254679, 151.223139], [34.940664, 34.987262, 35.029079, 35.041712, 35.054354]])], edgecolor='black', linestyle='dashed', linewidth=2)

#im.show_polygons([np.array([[151.983333333, 151.507416667, 151.421458333, 151.252875, 152.017958333], [34.724166667, 34.974416667, 34.995083333, 35.289305556, 35.278805556]]), np.array([[152.019166667, 151.86525, 151.522875, 151.183375, 151.183833333, 151.070833333, 151.075916667, 152.029958333], [34.578472222, 34.646472222, 34.848611111, 34.980055556, 35.219027778, 35.214527778, 34.506166667, 34.508444444]])], facecolor='white', zorder=2)

#im.show_ellipses(np.array([151.483531]), np.array([35.046612]), 0.1, 0.15, facecolor='black')

######
'''

'''
im = aplpy.FITSFigure(path + lofar_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.15, 0.1, 0.75, 0.8])
im_1 = aplpy.FITSFigure(path_hires + inset_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.16, 0.59, 0.35, 0.30])
im_2 = aplpy.FITSFigure(path_hires + inset_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.54, 0.11, 0.35, 0.30])

im.recenter(151.55, 34.90, radius=0.3)
#im.show_colorscale(cmap='jet', stretch='linear', vmin=-1.6, vmax=-0.4)

im_1.recenter(151.28584, 35.02501, radius=0.065)
#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=-1.6, vmax=-0.4)
#im_1.show_colorscale(cmap='jet', stretch='log', vmin=1e-5, vmax=1e-1)
#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=0, vmax=1)

im_1.show_grayscale(invert=True, stretch='log', vmin=0.001, vmax=0.1)
#im_1.show_grayscale(invert=True, stretch='log', vmin=0.001, vmax=0.01)
#im_1.axis_labels.set_font(size=18)
#im_1.tick_labels.set_font(size=16)

im_2.recenter(151.87969, 34.700476, radius=0.065)
#im_2.show_colorscale(cmap='jet', stretch='linear', vmin=-1.6, vmax=-0.4)
#im_1.show_colorscale(cmap='jet', stretch='log', vmin=1e-5, vmax=1e-1)
#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=0, vmax=1)

#im_2.show_grayscale(invert=False, stretch='linear', vmin=0.001, vmax=0.06)
im_2.show_grayscale(invert=True, stretch='log', vmin=0.001, vmax=0.5)
#im_2.axis_labels.set_font(size=18)
#im_2.tick_labels.set_font(size=16)

#im.show_colorscale(cmap='jet', stretch='linear', vmin=0., vmax=1.)

#im.show_grayscale(invert=False, stretch='linear', vmin=-0.001, vmax=0.07)
im.show_grayscale(invert=True, stretch='linear', vmin=0.000001, vmax=0.15)
im.axis_labels.set_font(size=18)
im.tick_labels.set_font(size=16)
'''
#'''
# Spix maps
im = aplpy.FITSFigure(path + lofar_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.15, 0.1, 0.8, 0.8])
#im_1 = aplpy.FITSFigure(path + inset_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.23, 0.62, 0.27, 0.27])
#im_2 = aplpy.FITSFigure(path + inset_im, figure=fig, dimensions=[0, 1], slices=[0, 0], subplot=[0.56, 0.11, 0.30, 0.25])

im.recenter(151.55, 34.90, radius=0.32)
im.show_colorscale(cmap='jet', stretch='linear', vmin=-1.1, vmax=-0.4)
#im.show_colorscale(cmap='jet', stretch='linear', vmin=0, vmax=1)
#im.show_colorscale(cmap='jet', stretch='linear', vmin=0.1, vmax=0.25)

#im_1.recenter(151.28584, 35.02501, radius=0.07)
#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=-1.6, vmax=-0.4)
#im_1.show_colorscale(cmap='jet', stretch='log', vmin=1e-5, vmax=1e-1)
#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=0, vmax=1)
#im_1.show_colorscale(cmap='jet', stretch='linear', vmin=0.1, vmax=0.25)
#im_1.axis_labels.set_font(size=18)
#im_1.tick_labels.set_font(size=16)

#im_2.recenter(151.87969, 34.700476, radius=0.07)
#im_2.show_colorscale(cmap='jet', stretch='linear', vmin=-1.6, vmax=-0.4)
#im_1.show_colorscale(cmap='jet', stretch='log', vmin=1e-5, vmax=1e-1)
#im_2.show_colorscale(cmap='jet', stretch='linear', vmin=0, vmax=1)
#im_2.show_colorscale(cmap='jet', stretch='linear', vmin=0.1, vmax=0.25)
#im_2.axis_labels.set_font(size=18)
#im_2.tick_labels.set_font(size=16)

#im.show_colorscale(cmap='jet', stretch='linear', vmin=0., vmax=1.)
#im.show_grayscale(invert=True, stretch='linear', vmin=-0.01, vmax=0.1)
#im.axis_labels.set_font(size=18)
#im.tick_labels.set_font(size=16)

im.add_colorbar()
im.colorbar.show()
im.colorbar.set_location('right')
im.colorbar.set_width(0.1)
im.colorbar.set_pad(0.03)
im.colorbar.set_axis_label_text(r'$\alpha_{143}^{609}$')
##im.colorbar.set_axis_label_text(r'$\alpha_{143}^{609}$ - $\alpha_{609}^{1400}$ spectral curvature' )
##im.colorbar.set_axis_label_text('Spectral curvature error')
##im.colorbar.set_axis_label_text('Spectral index error')
im.colorbar.set_axis_label_font(size=20)
im.colorbar.set_axis_label_pad(20)
im.colorbar.set_font(size=16)

#im_1.add_colorbar()
#im_1.colorbar.show(log_format=True)
#im_1.colorbar.set_location('right')
#im_1.colorbar.set_width(0.1)
#im_1.colorbar.set_pad(0.03)
#im_1.colorbar.set_axis_label_font(size=16)
#im_1.colorbar.set_font(size=13)
#im_1.colorbar.set_ticks([1e-2,2e-2])
#im_1.set_title('Spectral index error')
#im_1.set_tick_color('k')
#'''
'''
im.add_scalebar(0.2)
im.scalebar.show(0.2)  # length in degrees
from astropy import units as u
im.scalebar.set_length(8.721 * u.arcminute)
im.scalebar.set_label('5 pc')
im.scalebar.set_corner('top right')
im.scalebar.show(8.721 * u.arcminute, label="1 Mpc", corner="top right", color='black')

im.add_beam()
im.beam.set_edgecolor('white')
im.beam.set_facecolor('black')
im.beam.set_hatch('/')
##im.set_theme('publication')
im.tick_labels.set_xformat('hh:mm:ss')
im.tick_labels.set_yformat('dd:mm')
im.set_tick_color('k')
im.set_nan_color('white')
#im.axis_labels.set_font(size=18)
#im.tick_labels.set_font(size=16)

im_1.add_beam()
im_1.beam.set_edgecolor('white')
im_1.beam.set_facecolor('black')
im_1.beam.set_hatch('/')
im_1.set_tick_color('k')
aplpy.Ticks(im_1).hide_y()
aplpy.TickLabels(im_1).hide_y()
im_1.axis_labels.hide()
im_1.tick_labels.set_xformat('hh:mm')
im_1.set_nan_color('black')

im_2.add_beam()
im_2.beam.set_edgecolor('white')
im_2.beam.set_facecolor('black')
im_2.beam.set_hatch('/')
im_2.set_tick_color('k')
aplpy.Ticks(im_2).hide_y()
aplpy.TickLabels(im_2).hide_y()
im_2.axis_labels.hide()
im_2.tick_labels.set_xformat('hh:mm')
im_2.tick_labels.set_xposition('top')
im_2.set_nan_color('black')
'''

#im.show_arrows(np.array([151.725, 151.579]), np.array([35., 34.69]), np.array([-0.3, 0.25]), np.array([0., 0.]), width=0.5, head_width=5., color='black')
#im.show_arrows(np.array([151.579]), np.array([34.69]), np.array([0.25]), np.array([0.]), width=0.5, head_width=5., color='black')

#im.show_ellipses(np.array([151.3083]), np.array([35.016]), np.array([0.05]), np.array([0.02]), angle=30, color='black', linestyle='--')

#im.show_circles(np.array([151.925, 151.854, 151.821, 151.383, 151.325]), np.array([34.65, 34.75, 34.75, 34.96, 35.05]), np.array([0.015, 0.015, 0.01, 0.01, 0.01]), facecolor='white', zorder=1000)
#im_1.show_circles(np.array([151.325]), np.array([35.05]), np.array([0.01]), facecolor='white', zorder=1000) 

# Manipulate subplot axes via the figure, tick label colors not acessible via AplPy FitsFigure

#fig.axes[1].tick_params(colors='black')
#fig.axes[1].spines['top'].set_color('black')
#fig.axes[1].spines['bottom'].set_color('black')
#fig.axes[1].spines['left'].set_color('black')
#fig.axes[1].spines['right'].set_color('black')
#fig.axes[2].tick_params(colors='black')
#fig.axes[2].spines['top'].set_color('black')
#fig.axes[2].spines['bottom'].set_color('black')
#fig.axes[2].spines['left'].set_color('black')
#fig.axes[2].spines['right'].set_color('black')
#'''
#plt.setp(plt.gca().get_xticklabels(), color='red')
#'''
factor = 5.
sigma = 0.6e-3
sigma1 = 0.5e-3
sigma_sm = 1.6e-3
sigma_spix = 3.e-3
sigma_ddf = 0.26e-3
#sigma_ddf_1 = 380.e-6
levno = 10
levbase = 2.
levels = np.power(np.sqrt(levbase), range(levno))
levels = np.insert(levels, 0, -levels[0], axis=0)
print ("Levels: ", factor * sigma_spix * levels)
im.show_contour(path+lofar_im, levels=factor * sigma_spix * levels, colors=['gray'], linewidths=0.5, overlap=True)
#im_1.show_contour(path_hires+inset_im, levels=factor * sigma_ddf * levels, colors=['gray'], linewidths=0.5)
#im_2.show_contour(path_hires+inset_im, levels=factor * sigma_ddf * levels, colors=['gray'], linewidths=0.5)
#'''

pos = SkyCoord(["10:05:01.451 +35:02:25.384", "10:05:18.036 +34:58:31.189", "10:05:21.152 +34:59:59.059", "10:05:25.165 +35:01:26.910", "10:05:39.715 +  34:57:23.725", "10:05:49.552 +34:54:03.678", "10:06:23.471 +34:51:07.540", "10:06:45.753 +34:47:56.557", "10:07:10.459 +34:44:55.307", "10:07:40.046 +  34:42:16.965", "10:07:38.235 +34:40:48.925"], frame=FK5, unit=(u.hourangle, u.degree))

im.show_rectangles(np.array(pos.ra.degree), np.array(pos.dec.degree), np.array([160.072, 60.6729, 71.6399, 71.6161, 99.0416, 87.9601, 87.6856, 109.442, 92.1936, 76.3952, 87.2109])/3600., np.array([156.318, 70.6044, 70.5003, 70.5303, 98.1844, 82.1228, 104.968, 111.356, 205.588, 62.6333, 84.9923])/3600., edgecolor='black', linewidth=1)

#im.show_rectangles(np.array([151.885108, 151.787620, 151.697969, 151.455176, 151.358325, 151.341375, 151.319955, 151.254662]), np.array([34.710602, 34.752080, 34.788003, 34.904841, 35.020029, 34.993528, 34.974135, 35.041590]), np.array([96.80, 79.56, 97.02, 92.67, 57.50, 56.80, 56.82, 93.13])/3600., np.array([76.52, 142.15, 94.78, 88.50, 79.49, 71.35, 75.56, 117.17])/3600., edgecolor='black', linewidth=1)

lin_SE = SkyCoord(["10:06:37.85571 +34:48:30.6278", "10:06:46.85853 +34:47:08.5337", "10:06:56.46752 +34:46:41.8236", "10:07:03.97766 +34:46:41.1244", "10:07:18.04733 +34:43:30.4109", "10:07:28.82627 +34:41:52.4792", "10:07:43.51362 +34:40:58.2728"], frame=FK5, unit=(u.hourangle, u.degree))

lin_NW = SkyCoord(["10:05:43.21060 +34:56:35.0412", "10:05:10.22578 +35:01:46.5361", "10:04:54.08773 +35:02:55.9048"], frame=FK5, unit=(u.hourangle, u.degree))

im.show_lines([np.array([lin_NW.ra.degree, lin_NW.dec.degree]), np.array([lin_SE.ra.degree, lin_SE.dec.degree])], edgecolor='black', linestyle='dashed', linewidth=1)

#im.show_lines([np.array([[151.438500,151.359650,151.294741,151.252644,151.221930], [34.940965,34.987429,35.028743,35.042122,35.054816]]), np.array([[151.650049,151.721693,151.771522,151.800626,151.830863,151.868466,151.930297], [34.814092,34.771757,34.768451,34.749226,34.712928,34.700573,34.678900]])], edgecolor='black', linestyle='dashed', linewidth=1)

im.show_circles(np.array([151.925, 151.854, 151.821, 151.383, 151.325]), np.array([34.65, 34.75, 34.75, 34.96, 35.05]), np.array([0.02, 0.015, 0.01, 0.01, 0.01]), facecolor='white', zorder=1000)
#im_1.show_circles(np.array([151.925, 151.854, 151.821, 151.383, 151.325]), np.array([34.65, 34.75, 34.75, 34.96, 35.05]), np.array([0.015, 0.015, 0.01, 0.01, 0.01]), facecolor='white', zorder=1000)
#im_2.show_circles(np.array([151.925, 151.854, 151.821, 151.383, 151.325]), np.array([34.65, 34.75, 34.75, 34.96, 35.05]), np.array([0.015, 0.015, 0.01, 0.01, 0.01]), facecolor='white', zorder=1000)

lab = SkyCoord(["10:07:40.36 +34:43:33.3", "10:07:35.35 +34:39:16.7", "10:07:11.86 +34:47:28.4", "10:06:47.72 +34:49:38.9", "10:06:23.05 +34:52:39.7", "10:05:49.5 +34:52:56.0", "10:05:38.05 +34:55:45.3", "10:05:26.55 +35:02:36.7", "10:05:26.57 +35:00:00.4", "10:05:17.48 +34:57:16.9", "10:05:00.73 +35:04:55.1", "10:07:39 +34:41:49.2", "10:07:37 +34:41:07.7", "10:07:37 +34:40:43.5"], frame=FK5, unit=(u.hourangle, u.degree))
labdegra = np.array(lab.ra.degree)
labdegdec = np.array(lab.dec.degree)

im.add_label(labdegra[0], labdegdec[0], text='1', size='large')
im.add_label(labdegra[1], labdegdec[1], text='2', size='large')
im.add_label(labdegra[2], labdegdec[2], text='3', size='large')
im.add_label(labdegra[3], labdegdec[3], text='4', size='large')
im.add_label(labdegra[4], labdegdec[4], text='5', size='large')
im.add_label(labdegra[5], labdegdec[5], text='6', size='large')
im.add_label(labdegra[6], labdegdec[6], text='7', size='large')
im.add_label(labdegra[7], labdegdec[7], text='8', size='large')
im.add_label(labdegra[8], labdegdec[8], text='9', size='large')
im.add_label(labdegra[9], labdegdec[9], text='10', size='large')
im.add_label(labdegra[10], labdegdec[10], text='11', size='large')
#im_2.add_label(labdegra[11], labdegdec[11], text='< H1', size='small')
#im_2.add_label(labdegra[12], labdegdec[12], text='< H2', size='small')
#im_2.add_label(labdegra[13], labdegdec[13], text='< H3', size='small')

im.add_beam()
im.beam.set_edgecolor('white')
im.beam.set_facecolor('black')
im.beam.set_hatch('/')
im.set_nan_color('white')