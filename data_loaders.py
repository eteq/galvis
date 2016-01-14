import os
import re
from glob import glob

import numpy as np

from astropy import units as u
from astropy.table import Table, QTable
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, SphericalRepresentation, CartesianRepresentation, UnitSphericalRepresentation
from astropy.coordinates.angles import rotation_matrix

### ELVIS simulation loaders

def read_elvis_z0(fn):
    tab = QTable.read(fn, format='ascii.commented_header',data_start=0, header_start=1)

    col_name_re = re.compile(r'(.*?)(\(.*\))?$')
    for col in tab.columns.values():
        match = col_name_re.match(col.name)
        if match:
            nm, unit = match.groups()
            if nm != col.name:
                col.name = nm
            if unit is not None:
               col.unit = u.Unit(unit[1:-1]) # 1:-1 to get rid of the parenthesis

    return tab

def load_elvii(data_dir=os.path.abspath('elvis_data/'), isolated=False, inclhires=False):
    tables = {}

    fntoload = glob(os.path.join(data_dir, '*.txt'))
    for fn in fntoload:
        simname = os.path.split(fn)[-1][:-4]
        if simname.startswith('i'):
            if not isolated:
                continue
        else:
            if isolated:
                continue
        if not inclhires and 'HiRes' in fn:
            continue
        print('Loading', fn)
        tables[simname] = read_elvis_z0(fn)
    return tables

def add_oriented_radecs(elvis_tab, hostidx=0, targetidx=1,
                        target_coord=SkyCoord(0*u.deg, 0*u.deg),
                        earth_location=[0,0,0]*u.kpc, roll_angle=0*u.deg):
    """
    Computes a spherical coordinate system centered on the `hostidx` halo,
    re-oriented so that `targetidx` is at the `target_coord` coordinate
    location.

    Note that this adds columns 'host<n>_lat', 'host<n>_lon', and 'host<n>_dist' to
    `elvis_tab`, and will *overwrite* them if  they already exist.
    """
    if hasattr(target_coord, 'spherical'):
        target_lat = target_coord.spherical.lat
        target_lon = target_coord.spherical.lon
    else:
        target_lat = target_coord.lat
        target_lon = target_coord.lon

    dx = u.Quantity((elvis_tab['X'])-elvis_tab['X'][hostidx]) + earth_location[0]
    dy = u.Quantity((elvis_tab['Y'])-elvis_tab['Y'][hostidx]) + earth_location[1]
    dz = u.Quantity((elvis_tab['Z'])-elvis_tab['Z'][hostidx]) + earth_location[2]

    cart = CartesianRepresentation(dx, dy, dz)
    sph = cart.represent_as(SphericalRepresentation)

    #first rotate the host to 0,0
    M1 = rotation_matrix(sph[targetidx].lon, 'z')
    M2 = rotation_matrix(-sph[targetidx].lat, 'y')
    #now rotate from origin to target lat,lon
    M3 = rotation_matrix(target_lat, 'y')
    M4 = rotation_matrix(-target_lon, 'z')
    #now compute any "roll" about the final axis
    targ_cart = UnitSphericalRepresentation(lat=target_lat, lon=target_lon).represent_as(CartesianRepresentation)
    M5 = rotation_matrix(roll_angle, targ_cart.xyz.value)

    M = (M5*M4*M3*M2*M1).A
    newxyz = np.dot(M, cart.xyz)

    cart2 = CartesianRepresentation(newxyz)
    sph2 = cart2.represent_as(SphericalRepresentation)

    elvis_tab['host{}_lat'.format(hostidx)] = sph2.lat.to(u.deg)
    elvis_tab['host{}_lon'.format(hostidx)] = sph2.lon.to(u.deg)
    elvis_tab['host{}_dist'.format(hostidx)] = sph2.distance


### GALFA-related loaders
def load_galfa_sensitivity(fn, rngscs=None, cendist=None):
    """
    `rngscs` is a length-two SkyCoords with the upper/lower bounds of a
    square cutoff

    `cendist` is a 2-tuple of (centerSkyCoord, angle) to cut out *only* the

    returns data, skycoords, wcs, hdu
    """
    f = fits.open(fn)
    hdu = f[0]

    wcs = WCS(hdu.header)

    # create the SkyCoord
    xp, yp = np.mgrid[:hdu.data.shape[1], :hdu.data.shape[0]]
    scs = SkyCoord.from_pixel(xp, yp, wcs)
    data = hdu.data.T

    if rngscs is not None:
        ra_arr = scs[:, 0].ra.value
        ra_arr[np.isnan(ra_arr)] = 1000
        dec_arr = scs[1000].dec.value  # 0 index is nans

        minra_idx = np.argmin(np.abs(ra_arr - np.min(rngscs.ra).deg))
        maxra_idx = np.argmin(np.abs(ra_arr - np.max(rngscs.ra).deg))
        ra_idxmin, ra_idxmax = min(minra_idx, maxra_idx), max(minra_idx, maxra_idx)
        mindec_idx = np.argmin(np.abs(dec_arr - np.min(rngscs.dec).deg))
        maxdec_idx = np.argmin(np.abs(dec_arr - np.max(rngscs.dec).deg))
        dec_idxmin, dec_idxmax = min(mindec_idx, maxdec_idx), max(mindec_idx, maxdec_idx)

        slc = (slice(ra_idxmin, ra_idxmax), slice(dec_idxmin, dec_idxmax))

        data = data[slc]
        scs = scs[slc]

    if cendist is not None:
        censc, dist = cendist
        seps = censc.separation(scs)
        sepmsk = seps <= dist

        data = data[sepmsk]
        scs = scs[sepmsk]

    qdata = data * u.Unit(hdu.header['BUNIT'].replace('seconds', 'second'))

    return qdata, scs, wcs, hdu
