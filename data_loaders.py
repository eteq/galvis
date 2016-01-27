import os
import re
from glob import glob

import numpy as np

from astropy import units as u
from astropy.table import QTable
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, SphericalRepresentation, CartesianRepresentation, UnitSphericalRepresentation
from astropy.coordinates.angles import rotation_matrix

### ELVIS simulation loaders


def read_elvis_z0(fn):
    tab = QTable.read(fn, format='ascii.commented_header', data_start=0, header_start=1)

    col_name_re = re.compile(r'(.*?)(\(.*\))?$')
    for col in tab.columns.values():
        match = col_name_re.match(col.name)
        if match:
            nm, unit = match.groups()
            if nm != col.name:
                col.name = nm
            if unit is not None:
                col.unit = u.Unit(unit[1:-1])  # 1:-1 to get rid of the parenthesis

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


galactic_center = SkyCoord(0*u.deg, 0*u.deg, frame='galactic')

def add_oriented_radecs(elvis_tab, hostidx=0, targetidx=1,
                        target_coord=SkyCoord(0*u.deg, 0*u.deg),
                        earth_distance=8.5*u.kpc, earth_vrot=220*u.km/u.s,
                        roll_angle=0*u.deg):
    """
    Computes a spherical coordinate system centered on the `hostidx` halo,
    re-oriented so that `targetidx` is at the `target_coord` coordinate
    location.

    Note that this adds columns 'host<n>_*' to
    `elvis_tab`, and will *overwrite* them if  they already exist.
    """
    if hasattr(target_coord, 'spherical'):
        target_lat = target_coord.spherical.lat
        target_lon = target_coord.spherical.lon
    else:
        target_lat = target_coord.lat
        target_lon = target_coord.lon

    def offset_repr(rep, vector, newrep=None):
        if newrep is None:
            newrep = rep.__class__
        newxyz = rep.to_cartesian().xyz + vector.reshape(3, 1)
        return CartesianRepresentation(newxyz).represent_as(newrep)

    def rotate_repr(repr, matrix, newrep=None):
        if newrep is None:
            newrep = rep.__class__
        newxyz = np.dot(matrix.view(np.ndarray), rep.to_cartesian().xyz)
        return CartesianRepresentation(newxyz).represent_as(newrep)

    rep = CartesianRepresentation(elvis_tab['X'], elvis_tab['Y'], elvis_tab['Z'])
    # first we offset the catalog to have its origin at host0
    rep = offset_repr(rep, -rep.xyz[:, hostidx])

    # now rotate so that host1 is along the z-axis, and apply the arbitrary roll angle
    usph = rep.represent_as(UnitSphericalRepresentation)
    M1 = rotation_matrix(usph.lon[targetidx], 'z')
    M2 = rotation_matrix(90*u.deg-usph.lat[targetidx], 'y')
    M3 = rotation_matrix(roll_angle, 'z')
    rep = rotate_repr(rep, M3*M2*M1)

    # now determine the location of the earth in this system
    target_gc_angle = target_coord.separation(galactic_center)
    target_distance = rep.z[targetidx]  # distance to the target host
    # law of sines formula applied to SSA triangle
    sphi = np.sin(target_gc_angle + np.arcsin(earth_distance*np.sin(target_gc_angle)/target_distance))
    earth_location = u.Quantity([earth_distance * sphi,
                                 0*u.kpc,
                                 earth_distance * (1-sphi**2)**0.5])  # cos(arcsin(sphi))

    # now offset to put earth at the origin
    rep = offset_repr(rep, earth_location)
    sph = rep.represent_as(SphericalRepresentation)

    # rotate to put the target at its correct spot
    # first sent the target host to 0,0
    M1 = rotation_matrix(sph[targetidx].lon, 'z')
    M2 = rotation_matrix(-sph[targetidx].lat, 'y')
    # now rotate from origin to target lat,lon
    M3 = rotation_matrix(target_lat, 'y')
    M4 = rotation_matrix(-target_lon, 'z')

    rep = rotate_repr(rep, M4*M3*M2*M1)

    sph = rep.represent_as(SphericalRepresentation)
    elvis_tab['host{}_lat'.format(hostidx)] = sph.lat.to(u.deg)
    elvis_tab['host{}_lon'.format(hostidx)] = sph.lon.to(u.deg)
    elvis_tab['host{}_dist'.format(hostidx)] = sph.distance
    return

    # now compute  velocities
    # host galactocentric
    dvxg = u.Quantity((elvis_tab['Vx'])-elvis_tab['Vx'][hostidx])
    dvyg = u.Quantity((elvis_tab['Vy'])-elvis_tab['Vy'][hostidx])
    dvzg = u.Quantity((elvis_tab['Vz'])-elvis_tab['Vz'][hostidx])
    dxg = dx - earth_location[0]
    dyg = dy - earth_location[1]
    dzg = dz - earth_location[2]
    vrg = (dvxg*dxg + dvyg*dyg + dvzg*dzg) * (dxg**2+dyg**2+dzg**2)**-0.5
    elvis_tab['host{}_galvr'.format(hostidx)] = vrg.to(u.km/u.s)

    # "vLSR-like"
    earth_rotdir = SkyCoord(90*u.deg, 0*u.deg, frame='galactic').icrs
    vxyz = earth_rotdir.cartesian.xyz * earth_vrot
    dvx = dvxg + vxyz[0]
    dvy = dvyg + vxyz[1]
    dvz = dvzg + vxyz[2]

    vrlsr = (dvx*cart2.x + dvy*cart2.y + dvz*cart2.z) * (cart2.x**2+cart2.y**2+cart2.z**2)**-0.5
    elvis_tab['host{}_vrlsr'.format(hostidx)] = vrlsr.to(u.km/u.s)

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

    #qdata = data * u.Unit(hdu.header['BUNIT'].replace('seconds', 'second'))
    # the unit in the header is wrong - should be solar masses @ 1 Mpc
    qdata = data * u.Msun * u.Mpc**-2

    return qdata, scs, wcs, hdu
