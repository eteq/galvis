import os
import re
from glob import glob

import numpy as np
from scipy import optimize

from astropy import units as u
from astropy.table import QTable, Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import (SkyCoord, ICRS, SphericalRepresentation,
                                 CartesianRepresentation,
                                 UnitSphericalRepresentation, Distance, Angle)

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


def load_elvii_z0(data_dir=os.path.abspath('elvis_data/PairedCatalogs/'), isolated=False, inclhires=False):
    tables = {}

    fntoload = [fn for fn in glob(os.path.join(data_dir, '*.txt')) if 'README' not in fn]
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
        tables[simname] = tab = read_elvis_z0(fn)
        annotate_table_z0(tab)
    return tables


def annotate_table_z0(tab):
    id0, id1 = tab['ID'][:2]
    tab['sat_of_0'] = tab['UpID'] == id0
    tab['sat_of_1'] = tab['UpID'] == id1
    tab['sat_of_either'] = tab['sat_of_0']|tab['sat_of_1']


def read_elvis_trees(dirfn, cols):
    if isinstance(cols, str):
        cols = cols.split(',')

    coldct = {}
    for col in cols:
        fn = os.path.join(dirfn, col) + '.txt'
        with open(fn) as f:
            firstline = f.readline().strip()
            secondline = f.readline().strip()
        dt = float if '.' in secondline else int
        arr = np.loadtxt(fn, dtype=dt)

        match = re.match(r'# .*?\((.*?)\)', firstline)
        unit = u.Unit(match.group(1))  if match else None

        coldct[col] = arr*(1 if unit is None else unit)

    if 'scale' in coldct:
        coldct['z'] = 1./coldct['scale'] - 1
    return QTable(coldct)


def load_elvii_trees(cols, data_dir=os.path.abspath('elvis_data/PairedTrees/'), isolated=False, inclhires=False):
    tables = {}

    fntoload = [fn for fn in glob(os.path.join(data_dir, '*')) if os.path.isdir(fn)]

    for fn in fntoload:
        simname = os.path.split(fn)[-1]
        if simname.startswith('i'):
            if not isolated:
                continue
        else:
            if isolated:
                continue
        if not inclhires and 'HiRes' in fn:
            continue
        print('Loading', fn)
        tables[simname] = read_elvis_trees(fn, cols)
    return tables

def annotate_z0_from_trees(tab0s, tree_tabs, zthresh=None):
    """
    Updates the ``tab0s`` catalogs based on info from walking the ``tree_tabs``.
    `zthresh` sets what "z since" to use, or None for "ever"

    Updates the catalogs to include columns for:
    * upID<x>0
    * upID<x>1
    * upID<x>_either
    where <x> is 'ever' or z<n>
     """
    for nm in tab0s:
        tab0 = tab0s[nm]
        trees = tree_tabs[nm]

        up0 = (trees['ID'][0] == trees['upID'])
        up0[(trees['ID'][0]==0) | (trees['upID']==0)] = False
        up1 = trees['ID'][1] == trees['upID']
        up1[(trees['ID'][1]==0) | (trees['upID']==0)] = False

        if zthresh is not None:
            up0[trees['z'] > zthresh] = False
            up1[trees['z'] > zthresh] = False

        ever0 = np.any(up0, axis=1)
        ever1 = np.any(up1, axis=1)

        basename = 'upID' + ('ever' if zthresh is None else 'z{:.2g}'.format(zthresh).replace('.', 'p'))
        tab0[basename + '0'] = ever0
        tab0[basename + '1'] = ever1
        tab0[basename + '_either'] = ever0 | ever1


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

    def rotate_repr(rep, matrix, newrep=None):
        if newrep is None:
            newrep = rep.__class__
        newxyz = np.dot(matrix.view(np.ndarray), rep.to_cartesian().xyz)
        return CartesianRepresentation(newxyz).represent_as(newrep)

    rep = CartesianRepresentation(elvis_tab['X'], elvis_tab['Y'], elvis_tab['Z'])
    # first we offset the catalog to have its origin at host
    rep = offset_repr(rep, -rep.xyz[:, hostidx])

    # now rotate so that host1 is along the z-axis, and apply the arbitrary roll angle
    usph = rep.represent_as(UnitSphericalRepresentation)
    M1 = rotation_matrix(usph.lon[targetidx], 'z')
    M2 = rotation_matrix(90*u.deg-usph.lat[targetidx], 'y')
    M3 = rotation_matrix(roll_angle, 'z')
    Mfirst = M3*M2*M1
    rep = rotate_repr(rep, Mfirst)

    # now determine the location of the earth in this system
    # need diagram to explain this, but it uses SSA formula
    theta = target_coord.separation(galactic_center)  # target to GC angle
    D = rep.z[targetidx]  # distance to the target host
    R = earth_distance
    # srho = (R/D) * np.sin(theta)
    # sdelta_p = (srho * np.cos(theta) + (1 - srho**2)**0.5)
    # sdelta_m = (srho * np.cos(theta) - (1 - srho**2)**0.5)
    d1, d2 = R * np.cos(theta), (D**2 - (R * np.sin(theta))**2)**0.5
    dp, dm = d1 + d2, d1 - d2
    sdelta = (dp/D) * np.sin(theta)

    x = R * sdelta
    z = R * (1-sdelta**2)**0.5
    earth_location = u.Quantity([x, 0*u.kpc, z])

    # now offset to put earth at the origin
    rep = offset_repr(rep, -earth_location)
    sph = rep.represent_as(SphericalRepresentation)

    # rotate to put the target at its correct spot
    # first sent the target host to 0,0
    M1 = rotation_matrix(sph[targetidx].lon, 'z')
    M2 = rotation_matrix(-sph[targetidx].lat, 'y')
    # now rotate from origin to target lat,lon
    M3 = rotation_matrix(target_lat, 'y')
    M4 = rotation_matrix(-target_lon, 'z')
    Mmiddle = M4*M3*M2*M1
    rep = rotate_repr(rep, Mmiddle)

    # now one more rotation about the target to stick the GC in the right place
    def tomin(ang, inrep=rep[hostidx], axis=rep[targetidx].xyz, target=galactic_center.icrs):
        newr = rotate_repr(inrep, rotation_matrix(ang[0]*u.deg, axis))
        return ICRS(newr).separation(target).radian
    rot_angle = optimize.minimize(tomin, np.array(0).ravel(), method='Nelder-Mead')['x'][0]
    Mlast = rotation_matrix(rot_angle*u.deg, rep[targetidx].xyz)
    rep = rotate_repr(rep, Mlast)

    sph = rep.represent_as(SphericalRepresentation)
    elvis_tab['host{}_lat'.format(hostidx)] = sph.lat.to(u.deg)
    elvis_tab['host{}_lon'.format(hostidx)] = sph.lon.to(u.deg)
    elvis_tab['host{}_dist'.format(hostidx)] = sph.distance

    # now compute  velocities
    # host galactocentric
    dvxg = u.Quantity((elvis_tab['Vx'])-elvis_tab['Vx'][hostidx])
    dvyg = u.Quantity((elvis_tab['Vy'])-elvis_tab['Vy'][hostidx])
    dvzg = u.Quantity((elvis_tab['Vz'])-elvis_tab['Vz'][hostidx])

    earth_location_in_xyz = np.dot(Mfirst.T, earth_location)
    dxg = elvis_tab['X'] - elvis_tab['X'][0] - earth_location_in_xyz[0]
    dyg = elvis_tab['Y'] - elvis_tab['Y'][0] - earth_location_in_xyz[0]
    dzg = elvis_tab['Z'] - elvis_tab['Z'][0] - earth_location_in_xyz[0]
    vrg = (dvxg*dxg + dvyg*dyg + dvzg*dzg) * (dxg**2+dyg**2+dzg**2)**-0.5
    elvis_tab['host{}_galvr'.format(hostidx)] = vrg.to(u.km/u.s)

    # "vLSR-like"
    # first figure out the rotation direction
    earth_rotdir = SkyCoord(90*u.deg, 0*u.deg, frame='galactic').icrs

    #now apply the component from that everywhere
    offset_angle = earth_rotdir.separation(ICRS(sph))
    vrlsr = vrg - earth_vrot*np.cos(offset_angle)

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

def load_mccon12_table(mcconn_url='https://www.astrosci.ca/users/alan/Nearby_Dwarfs_Database_files/NearbyGalaxies.dat', qtable=True):
    from astropy.utils import data
    from astropy.io import ascii

    # have to do SSL stuff because astrosci.ca has expired SSL
    import ssl
    from urllib.error import URLError

    baseline_create = ssl._create_default_https_context
    try:
        mcconn_tab_str = data.get_file_contents(mcconn_url, cache=True)
    except URLError as e:
        ee[0] = e
        if 'SSL: CERTIFICATE_VERIFY_FAILED' in str(e.args):
            ssl._create_default_https_context = ssl._create_unverified_context
            exec(toexec)
        else:
            raise
    finally:
        ssl._create_default_https_context = baseline_create


    headerrow = mcconn_tab_str.split('\n')[32]
    colnames = headerrow.split()
    colidxs = [headerrow.rindex(col) for col in colnames]

    # this *removes* the references
    col_starts = colidxs[:-1]
    col_ends = [i-1 for i in colidxs[1:]]
    colnames = colnames[:-1]

    str_tab = ascii.read(mcconn_tab_str.split('\n')[34:], format='fixed_width_no_header',
                         names=colnames, col_starts=col_starts, col_ends=col_ends)

    mcconn_tab = (QTable if qtable else Table)()
    mcconn_tab['Name'] = [s.strip() for s in str_tab['GalaxyName']]

    scs = []
    for row in str_tab:
        dm = float(row['(m-M)o'].split()[0])
        scs.append(SkyCoord(row['RA'], row['Dec'], unit=(u.hour, u.deg),
                            distance=Distance(distmod=dm)))
    mcconn_tab['Coords'] = SkyCoord(scs)

    for col in str_tab.colnames[3:]:
        if col in ('EB-V', 'F', 'MHI'):
            #single number
            mcconn_tab[col] = [float(s) for s in str_tab[col]]
        else:
            # num + -
            vals, ps, ms = [], [], []
            for s in str_tab[col]:
                val, p, m = s.split()
                vals.append(float(val))
                ps.append(float(p))
                ms.append(float(m))
            mcconn_tab[col] = vals
            mcconn_tab[col + '+'] = ps
            mcconn_tab[col + '-'] = ms

    return mcconn_tab
