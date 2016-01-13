import os
import re
from glob import glob

import numpy as np

from astropy import units as u
from astropy.table import Table, QTable
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

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

def load_elvii(data_dir=os.path.abspath('elvis_data/'), isolated=False):
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
        print('Loading', fn)
        tables[simname] = read_elvis_z0(fn)
    return tables


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
