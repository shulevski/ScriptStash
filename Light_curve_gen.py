'''
Populates a dataframe with extracted flux density values
measured usin Pyse at fixed positions taken from a .fits catalog
Measurements made in an annular region aroind zenith.

v0.1 - Created August 2019 by API ASPIRE summer student Chileshe Mutale
v0.2 - Edited by Aleksandar Shulevski
'''
import os
import sys
import csv
import glob
from datetime import datetime
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.transforms as transf
import matplotlib.dates as mdates
import astropy.io.fits as fits
import argparse
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.time import Time
import scipy.signal as sig
#import astropy.stats as stat
from astropy.visualization import hist
from astropy.modeling import models, fitting

#---------The configuration pyse uses to come up with the source, with respective arguments set.

def get_configuration():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--indir', type=str, default="/tmp",
                        help="Input directory for fitsfile.")
    parser.add_argument('--fitsfile', type=str, default="./",
                       help="Target fits file.")

    parser.add_argument('--threshold', type=float, default=5000,
                       help="RMS Threshold to reject image.")
    parser.add_argument('--outdir', type=str, default="./",
                       help="Desitnation directory.")
    parser.add_argument("--detection", default=5, type=float,
                        help="Detection threshold")
    parser.add_argument("--analysis", default=3, type=float,
                        help="Analysis threshold")
    parser.add_argument("--radius", default=0, type=float,
                            help="Radius of usable portion of image (in pixels)")
    parser.add_argument("--grid", default=64, type=float,
                            help="Background grid segment size")
    parser.add_argument("--reference", default="", type=str,
                            help="Path of reference catalogue used for flux fitting.")
    parser.add_argument("--chunk_id", default="0", type=int,
                            help="ID of A12 data chunk.")  
    return parser.parse_args()
#---------Inspects image of radius R, to determine positions of known co ordinates [Kept]
def get_positions(filename, coords, R):
    kept = []
    star_quant = []
    with fits.open(filename) as fimage:
         zra, zdec = fimage[0].header['CRVAL1'], fimage[0].header['CRVAL2']
    if zra < 0.:
        zra = zra + 360.
    zenith = SkyCoord(zra, zdec, frame='icrs', unit='deg')
    for i in coords:
        star = SkyCoord(i[0][0], i[0][1], frame='icrs', unit='deg')
        if zenith.separation(star).degree < R:
            #print ("For addition: ", coords.index(i))
            kept.append(i[0])
            star_quant.append((i[1], i[2]))
            print ("Adding: ", star , ' ', i[2], " with separation ", zenith.separation(star).degree, " degrees")
            print ("Kept: ", kept, "size: ", len(kept))
    return kept, star_quant

def create_data(imgsloc=None, incatloc=None, incatname=None, infiles=None, bmaj=None, bmin=None, outloc=None, outlocbad=None, outname=None, searchrad=None):
    from sourcefinder.accessors import open as open_accessor
    from sourcefinder.accessors import sourcefinder_image_from_accessor
    
    # ------ Defining varaibles and structures we will need later
    coords = []
    kept_created = False
    cfg=get_configuration()
    cfg.indir=imgsloc
    frame =pd.DataFrame(columns=['RA','DEC','FLUX','FLUX_ERROR','TIME','STAR_ID', 'STAR_NAME'])
    # -------
    
    configuration = {
                    "back_size_x": cfg.grid,
                    "back_size_y": cfg.grid,
                    "margin": 0,
                    "radius": cfg.radius}
    
    catab=Table.read(incatloc+incatname+'.fits', hdu=1)
    
    pancat = catab.to_pandas()
    #catab.info()
    for idx, row in pancat.iterrows():
        coords.append([(row['ra'], row['dec']), row['source_id'], row['common_name']])
    
    for file in sorted(glob.glob(cfg.indir+infiles+'.fits')):
        cfg.fitsfile=(file)
        if kept_created == False:
            kept, star_quant = get_positions(cfg.fitsfile, coords, searchrad)
            kept_created = True
        
        with fits.open(cfg.fitsfile, mode='update') as hdul:
            
            print(cfg.fitsfile)
            hdr = hdul[0].header 
            timestamp = (datetime.strptime(hdr['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"))
            #timestamp = hdr['DATE-OBS']
            hdr.update(TELESCOP='A12')
            print(hdr['BMAJ'])
            # Only put values manually here for dirty images, i.e. beam=[0, 0] !!!
            hdr.update(BMAJ=bmaj)
            hdr.update(BMIN=bmin)
            print(hdr['BMAJ'])
            hdul.flush()
            # ------------------------------------------------------------------#
            #print(timestamp)
            try:    
                imagedata = sourcefinder_image_from_accessor(open_accessor(hdul,plane=0),**configuration)
                            
                #pyse_out = imagedata.extract(det=cfg.detection, anl=cfg.analysis,
                #                               labelled_data=None, labels=[],
                #                               force_beam=True)
            
                pyse_out = imagedata.fit_fixed_positions(positions = kept,boxsize=20,threshold=None,fixed='position+shape')
            
            except Exception as exc:
                print("Bad file.")
                os.system('mv ' + cfg.fitsfile + ' ' + outlocbad)
                print(exc)
                continue
            #else:
            #    print("Good file.")
            disc=0
            for i in range(len(pyse_out)):
                #print('Coord err: ', abs((pyse_out[i].dec.value - kept[i+disc][1])))
                if abs((pyse_out[i].dec.value - kept[i+disc][1])) > 0.1:
                    disc+=1
                    print('Pyse missed to extract a source, skipping it in the list of targets for this image.')
                frame = frame.append({'RA' : ((pyse_out[i].ra.value)+360), 'DEC' : pyse_out[i].dec.value, 'FLUX' : pyse_out[i].flux.value, 'FLUX_ERROR' : pyse_out[i].flux.error,'TIME' : timestamp, 'STAR_ID' : star_quant[i+disc][0], 'STAR_NAME': star_quant[i+disc][1]}, ignore_index=True)
                #print(frame)
                #print('Kept RA, DEC: ', kept[i+disc])
                #print('Star quant: ', star_quant[i+disc])
                #print('Pyse out: ', pyse_out[i].dec.value)
                #print('Pyse out: ', pyse_out)
                #print('Kept: ', kept)
                #print('Length pyse out: ', len(pyse_out))
                #print('Length kept: ', len(kept))
                   
    #print(pyse_out[i].flux.value)
    #print(frame)
    #print("source_ID is",hdr['source_id'])
    
    frame.to_pickle(outloc + outname + '.pkl')
    print('Finished '+ outname)

#---------plots the light curve
def plot_data(lightcurve_file, incat_file):
     
    frame_hi = pd.read_pickle(lightcurve_file)
    #print(type(frame_hi['STAR_ID'][0]))
    print(frame_hi['STAR_ID'])
    catab=Table.read(incat_file, hdu=1)
    pancat = catab.to_pandas()
    for idx, row in pancat.iterrows():
        star_frame_hi = frame_hi.loc[frame_hi['STAR_ID'] == row['source_id']]
        print(str(row['source_id']))
        try:
            star_frame_hi.iloc[0]['STAR_NAME']
        except Exception as exc:
            print(str(row['common_name']) + ' was not observed.')
            continue
        fig, axis = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]})
        #star_frame_lo = frame_hi.loc[frame_lo['STAR_ID'] == star_id]
        print(str(star_frame_hi.iloc[0]['STAR_NAME']).strip())
        
        #plot_title = str(star_frame_hi.iloc[0]['STAR_NAME']).split("'")[1].strip()
        plot_title = str(star_frame_hi.iloc[0]['STAR_NAME']).strip()
        #star_frame_hi.plot(x='TIME', y='FLUX', yerr='FLUX_ERROR', ecolor='orange', title=plot_title+' 71MHz', ax=axis)
        #star_frame_hi.plot(x='TIME', y='FLUX', yerr='FLUX_ERROR', ecolor='orange', title=plot_title+' 71MHz', ax=axis)
        ylmts = 5
        #star_frame_hi['FLUX_MED'] = sig.medfilt(star_frame_hi['FLUX'], kernel_size=13)
        #hist = stat.histogram(star_frame_hi['FLUX_MED'], bins='scott')
        
        #star_frame_hi.plot(x='TIME', y='DEC', yerr='FLUX_ERROR', ecolor='orange', title=plot_title+' 61MHz', ax=axis)
        
        star_frame_hi.plot(x='TIME', y='FLUX', yerr='FLUX_ERROR', ecolor='orange', title=plot_title + ' 1m 61MHz', ax=axis[0])
        #bg_interval = star_frame_hi['TIME'] > '2019-12-22 22:46:20.0'
        #en_interval = star_frame_hi['TIME'] < '2019-12-22 22:46:40.0'
        #star_frame_hi.loc[bg_interval & en_interval].plot(x='TIME', y='FLUX_MED', yerr='FLUX_ERROR', 
        #                                                  ecolor='orange', title=plot_title+' 61MHz', ax=axis[0])
        
        #vals = star_frame_hi.loc[bg_interval & en_interval]['FLUX_MED']
        
        #axis[1].plot(hist[0])
        #print('Hist bins: ', hist[1])
        axis[1].set_ylim(bottom=-ylmts, top=ylmts)
        axis[1].set(xticks=[])
        axis[1].set(xticklabels=[])
        #axis[1].hist(star_frame_hi['FLUX_MED'])
        val, bins, patches = hist(star_frame_hi['FLUX'], bins='scott', ax=axis[1], density=True, 
histtype='stepfilled', alpha=0.3, orientation='horizontal')
        
        print('Len val: ', len(val))
        print('Len bins: ', len(bins[:len(bins)-1]))
        print('Len patches: ', len(patches))
        
        g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, bins[:len(bins)-1], val)
        
        print(g)
        base = axis[1].transData
        rot = transf.Affine2D().rotate_deg(-90)
        axis[1].plot(-bins[:len(bins)-1], g(bins[:len(bins)-1]), label='Gaussian fit', transform=rot+base)
       
        #star_frame_hi.plot(x='TIME', y='FLUX', yerr='FLUX_ERROR', ecolor='orange', title=plot_title+' 61MHz', ax=axis[0])
        #star_frame_lo.plot(x='TIME', y='FLUX', yerr='FLUX_ERROR', ecolor='orange', title=plot_title+' 22MHz', ax=axis[1])
        
        print('Star:', plot_title)
        print('RA: ', star_frame_hi.iloc[0]['RA']-360., 'DEC: ', star_frame_hi.iloc[0]['DEC'])
        print('TIME: ', star_frame_hi.iloc[0]['TIME'])
        
        #frame.loc[frame['DEC'] == 64.62615203857422].plot(x='TIME', y='FLUX',yerr='FLUX_ERROR')

        #frame.plot(x='TIME', y='FLUX', yerr='FLUX_ERROR')
        axis[0].set_ylabel('Stokes V Flux Density[Jy]')
        #axis.xaxis.set_major_locator(mdates.SecondLocator())
        #axis[0].xaxis.set_major_locator(mdates.MinuteLocator())
        axis[0].xaxis.set_major_locator(mdates.HourLocator())
        axis[0].xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM:%SS'))
        #axis.xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM'))
        axis[0].axhline(0.0, color='r', linestyle='--', lw=2)
        axis[0].set_ylim(bottom=-ylmts, top=ylmts)
        #axis.set_xlim(left='2018-12-15 03:53:00.000000', right='2018-12-15 03:54:00.000000')
        
        #axis[1].set_ylabel('Flux Density[Jy]')
        #axis[1].xaxis.set_major_locator(mdates.MinuteLocator())
        #axis[1].xaxis.set_major_formatter(mdates.DateFormatter('%HH:%MM:%SS'))
        #axis[1].axhline(0.0, color='r', linestyle='--', lw=2)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0)
        plt.savefig(plot_title+'.png')
        ##plt.hist(star_frame_hi['FLUX'][1000:], bins=500)
        ##plt.show()

def input_catalog():
    with fits.open('input_catalog.fits', memap=True, mode='update') as hdu_list:
        hdu_list.info()
        evt_data = Table(hdu_list[1].data)
        print (evt_data)
        print(hdu_list[1].columns)

def sandbox(path):
    for file in sorted(glob.glob(path+'*t00*V-image.fits')):
        with fits.open(file) as hdul:
            hdr = hdul[0].header
            timestamp = (datetime.strptime(hdr['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"))
            #timestamp = hdr['DATE-OBS']
            print (file)
            print (timestamp)
            print (timestamp.strftime('%H%M%S%f'))
            print (file.split('/')[8])
            print ((file.split('/')[8]).split('-')[3])
            os.system('cp '+path+file.split('/')[8]+' '+path+'/A12_'+timestamp.strftime('%H%M%S%f')+'-'+(file.split('/')[8]).split('-')[3]+'.fits')
    #os.system('rm '+path+'A12_high*.fits')

def custom_catalog():
    #event_coord = SkyCoord("8h39m57.600000000007725s", "-18d22m12.000000000003581s", frame='icrs', unit='deg') #GRB
    #event_coord = SkyCoord("13h47m15.74340s", "17d27m24.8552s", frame='icrs', unit='deg') # TauBoo
    #event_coord = SkyCoord("9h53m9.3s", "7d55m36.0s", frame='icrs', unit='deg') # PSR B0950+08
    #event_coord = SkyCoord("3h28m52.6s", "39d31m35.6s", frame='icrs', unit='deg') # Stokes V region transient
    event_coord = SkyCoord(["01h58m00.750s +65d43m00.31s", 
                            "08h09m28.800s +46d16m12.00s", 
                            "04h20m40.800s +73d37m48.00s", 
                            "10h36m45.600s +73d44m24.00s", 
                            "18h26m00.000s +81d24m00.00s", 
                            "05h31m58.000s +33d08m04.00s"], 
    frame='icrs', unit='deg'
    )
    ccat = Table()
    print(event_coord.ra.degree)
    print(event_coord.dec.degree)
    '''
    ccat['ra'] = [event_coord.ra.degree]
    ccat['dec'] = [event_coord.dec.degree]
    ccat['source_id'] = [333]
    ccat['common_name'] = ["22Dec2019"]
    '''
    ccat['ra'] = event_coord.ra.degree
    ccat['dec'] = event_coord.dec.degree
    ccat['source_id'] = [000, 111, 222, 333, 444, 555]
    ccat['common_name'] = ["FRB_20180916B", "FRB_20190907A", "FRB_20180814A", "FRB_20181030A", "FRB_20190212A", "FRB_20121102A"]
    ccat.write('/home/shulevski/Documents/Research/Projects/A12_LBA/Flare_dwarves/22Dec2019_A12_catalog_FRBs.fits', format='fits', 
overwrite=True)

def concat_data_frames(inpath, file_id):
    frames = [pd.read_pickle(f) for f in sorted(glob.glob(inpath+file_id+'*.pkl'))]
    final_df = frames[0].append(frames[1:], ignore_index=True)
    final_df.to_pickle(inpath+file_id+'_full.pkl')
    
if __name__ == '__main__':
    print ('Crunching...')
    '''
    cfg=get_configuration()
    create_data(imgsloc='/project/aartfaac/Data/image_frames/', \
            incatloc='/home/aartfaac-ashulevski/', \
            #incatname='PSR0950_A12_catalog', \
            incatname='input_catalog', \
            #incatname='TauBoo_A12_catalog', \
            #infiles='*I', \
            infiles='*V', \
            bmaj=0.304, #high \
            bmin=0.208, #high \
	    #bmaj=1.15, #low \
            #bmin=0.42, #low \
            outloc='/project/aartfaac/Data/Chunk_pickles/', \
            outlocbad='/project/aartfaac/Data/Chunk_pickles/bad_auto/', \
            #outname='PSR0950_A12_stokes_I', \
            #outname='GRB_A12_stokes_I', \
            outname='Star_flares_A12_stokes_V', \
            #outname='TauBoo_flare_A12_stokes_V', \
            searchrad=85.)
    '''
    #concat_data_frames('/project/aartfaac/Data/Chunk_pickles/','Star_flares_A12_stokes_V')
    #plot_data('/project/aartfaac/Data/Chunk_pickles/Star_flares_A12_stokes_V_full.pkl', '/home/aartfaac-ashulevski/input_catalog.fits')
    #input_catalog()
    #custom_catalog()
    #sandbox('/project/aartfaac/Data/chunk1/intervals/high/V/')
    '''
    for loc in sorted(glob.glob('/project/aartfaac/Data/chunk*')):
	 sandbox(loc+'/intervals/high/V/')
	 os.system('mv '+loc+'/intervals/high/V/' + 'A12_2*' + ' /project/aartfaac/Data/image_frames/')
	 cfg=get_configuration()
    	 create_data(imgsloc='/project/aartfaac/Data/image_frames/', \
            incatloc='/home/aartfaac-ashulevski/', \
            #incatname='PSR0950_A12_catalog', \
            incatname='input_catalog', \
            #incatname='TauBoo_A12_catalog', \
            #infiles='*I', \
            infiles='*V', \
            bmaj=0.304, #high \
            bmin=0.208, #high \
	    #bmaj=1.15, #low \
            #bmin=0.42, #low \
            outloc='/project/aartfaac/Data/Chunk_pickles/', \
            outlocbad='/project/aartfaac/Data/Chunk_pickles/bad_auto/', \
            #outname='PSR0950_A12_stokes_I', \
            #outname='GRB_A12_stokes_I', \
            outname='Star_flares_A12_stokes_V', \
            #outname='TauBoo_flare_A12_stokes_V', \
            searchrad=85.)
	 os.system('rm /project/aartfaac/Data/image_frames/*')
	 os.system('mv /project/aartfaac/Data/Chunk_pickles/Star_flares_A12_stokes_V.pkl /project/aartfaac/Data/Chunk_pickles/Star_flares_A12_stokes_V_' + loc.split("/")[4] + '.pkl')
    concat_data_frames('/project/aartfaac/Data/Chunk_pickles/','Star_flares_A12_stokes_V')
    '''
  
plot_data('/home/shulevski/Documents/Research/Projects/A12_LBA/Flare_dwarves/Results_20191222/1min/high_V/22Dec2019stokes_V_CRDra_full.pkl',
'/home/shulevski/Documents/Research/Projects/A12_LBA/Flare_dwarves/CRDra_A12_catalog.fits')
