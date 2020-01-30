#! /home/shulevski/miniconda2/envs/py27base/bin/python

def draw_altaz():
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.visualization import astropy_mpl_style
    plt.style.use(astropy_mpl_style)

    import astropy.units as u
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation, FK5

    A12 = EarthLocation(lat=52.91*u.deg, lon=6.87*u.deg, height=1*u.m)
    utcoffset = 0*u.hour  # UT
    time = Time('2019-2-19 01:35:00') - utcoffset


    #frame_July13night = AltAz(obstime=midnight+delta_midnight, location=A12)

    #m33altaz = m33.transform_to(AltAz(obstime=time,location=A12))
    #print("M33's Altitude = {0.alt:.2}".format(m33altaz))

    az = np.linspace(0., 360., 100)
    alt = np.ones(100) * 52.
    a = SkyCoord(np.array(az), np.array(alt), frame='altaz', unit='deg', obstime=time, location=A12)
    r = a.transform_to(FK5)

    print([x.ra.deg for x in r])
    print([x.dec.deg for x in r])

if __name__=='__main__':

    draw_altaz()