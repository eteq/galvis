import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord


def bradford_15_mstar_to_mgas(mstar):
    logx = np.log10(mstar/u.Msun)
    faintmsk = logx < 8.6
    mgas = np.empty_like(logx)
    mgas[faintmsk] = 1.052 * logx[faintmsk] + 0.236
    mgas[~faintmsk] = 0.461 * logx[~faintmsk] + 5.329
    return u.solMass*10**mgas


def add_scatter_to_elvis_mstar(elvii_pairs, scatter_amplitude=.3,
                               mstarcolnm='Mstar_preferred',
                               outcolnm='Mstar_scattered'):
    for nm, tab in elvii_pairs.items():
        lmstar = np.log10(tab[mstarcolnm]/u.solMass)
        lmstar_scat = lmstar + np.random.randn(len(tab))*scatter_amplitude
        tab['Mstar_scattered'] = (10**lmstar_scat)*u.solMass


def compute_elvis_mgas_bradford_15(elvii_pairs,
                                   mstarcolnm='Mstar_preferred',
                                   outcolnm='MHI'):
    for tab in elvii_pairs.values():
        Ms = tab[mstarcolnm]
        tab[outcolnm] = bradford_15_mstar_to_mgas(Ms)


def compute_detectability(sens_arr, sens_wcs, scs, MHIs):
    """
    Returns (nearest_sensitivity, in_survey, detectable), all arrays
    """
    pxs = scs.to_pixel(sens_wcs)
    xp = np.round(pxs[0]).astype(int)
    yp = np.round(pxs[1]).astype(int)
    msk = (0 <= xp)&(xp < sens_arr.shape[0])&(0 <= yp)&(yp < sens_arr.shape[1])

    sens = np.zeros(len(scs), dtype=sens_arr.dtype)*sens_arr.unit
    sens[msk] = sens_arr[xp[msk], yp[msk]]
    sens[sens == 0] = np.inf

    # test detectability
    det = sens * scs.distance**2 < MHIs

    return sens, np.isfinite(sens), det


def compute_elvis_detectability(sens_arr, sens_wcs, surveyname, elvii_pairs):
    for tab in elvii_pairs.values():
        for i in (0, 1):
            host_sc = SkyCoord(ra=tab['host{}_lon'.format(i)],
                               dec=tab['host{}_lat'.format(i)],
                               distance=tab['host{}_dist'.format(i)])
            detres = compute_detectability(sens_arr, sens_wcs, host_sc, tab['MHI'])

            # now fill in table columns
            tab['closest_sens_{}_host{}'.format(surveyname, i)] = detres[0]
            tab['in_survey_{}_host{}'.format(surveyname, i)] = detres[1]
            tab['detectable_{}_host{}'.format(surveyname, i)] = detres[2]


_vect_find_vlsr_minmax = None
def compute_vlsr_minmax(sc):
    global _vect_find_vlsr_minmax

    if _vect_find_vlsr_minmax is None:
        # this is adapted from a code by Yong Zheng
        from find_vlsr_minmax_allsky import find_vlsr_minmax  # accepts degrees l,b
        _vect_find_vlsr_minmax = np.vectorize(find_vlsr_minmax)

    scgal = sc.galactic
    return np.array(_vect_find_vlsr_minmax(scgal.l.degree, scgal.b.degree))


def compute_elvis_vlsr_minmax(elvii_pairs, verbose=False):
    for name, tab in elvii_pairs.items():
        for i in (0, 1):
            ra = tab['host{}_lon'.format(i)]
            dec = tab['host{}_lat'.format(i)]
            scs = SkyCoord(ra, dec)
            vdevmin, vdevmax = compute_vlsr_minmax(scs)*u.km/u.s
            vlsr = tab['host{}_vrlsr'.format(i)]
            tab['host{}_vdevok'.format(i)] = (vlsr > vdevmax) | (vdevmin > vlsr)
            if verbose:
                vdevok = tab['host{}_vdevok'.format(i)]
                print('Host', name.split('&')[i], 'frac vdev ok=',
                      np.sum(vdevok)/len(vdevok), 'of', len(scs))


def compute_hvc_d(scs, vlsrs, hvc_scs, hvc_vlsr):
    hvs_scs_repr = hvc_scs.data
    hvs_scs1_repr = hvc_scs.data.__class__(**{comp: getattr(hvs_scs_repr, comp).reshape(len(hvs_scs_repr), 1) for comp in hvs_scs_repr.components})
    hvc_scs1 = hvc_scs.realize_frame(hvs_scs1_repr)

    seps = scs.separation(hvc_scs1)  # this is now len(hvc_scs) x len(scs)
    dv = vlsrs - hvc_vlsr.reshape(len(hvc_vlsr), 1)
    return np.sqrt(seps.deg**2 + (dv.to('km/s').value/2.)**2)


def compute_elvis_hvc_ds(elvii_pairs, hvc_scs, hvc_vlsr, D_threshold=25, verbose=False):
    for name, tab in elvii_pairs.items():
        for i in (0, 1):
            scs = SkyCoord(tab['host{}_lon'.format(i)], tab['host{}_lat'.format(i)])
            vlsrs = tab['host{}_vrlsr'.format(i)]
            dHVCs = compute_hvc_d(scs, vlsrs, hvc_scs, hvc_vlsr)

            tab['host{}_dHVC'.format(i)] = dHVCs = np.min(dHVCs, axis=0)
            tab['host{}_dHVCok'.format(i)] = dHVCs > D_threshold
            if verbose:
                print('Host', name.split('&')[i], 'frac dHVC ok=',
                      np.sum(dHVCs > 25)/len(dHVCs), 'of', len(scs))


def compute_elvis_findable(elvii_pairs, hvc_scs, hvc_vlsr, D_threshold=25, verbose=False, vlsr_thresh=None):
    """
    Detectability must already be defined
    """
    recalc_vdev = recalc_dhvc = recalc_vlsr = False
    for name, tab in elvii_pairs.items():
        for i in (0, 1):
            if 'host{}_vdevok'.format(i) not in tab.colnames:
                recalc_vdev = True
            if 'host{}_dHVCok'.format(i) not in tab.colnames:
                recalc_dhvc = True
            if vlsr_thresh is not None and 'host{}_vlsrok'.format(i) not in tab.colnames:
                recalc_vlsr = True

    if recalc_vdev:
        compute_elvis_vlsr_minmax(elvii_pairs, verbose)
    if recalc_dhvc:
        compute_elvis_hvc_ds(elvii_pairs, hvc_scs, hvc_vlsr, D_threshold, verbose)
    if recalc_vlsr:
        for name, tab in elvii_pairs.items():
            for i in (0, 1):
                vlsr = tab['host{}_vrlsr'.format(i)]
                tab['host{}_vlsr{}ok'.format(i, vlsr_thresh.to(u.km/u.s).value)] = np.abs(vlsr) < vlsr_thresh


    for name, tab in elvii_pairs.items():
        for nm in tab.colnames:
            if nm.startswith('detectable'):
                suffix = nm[10:]
                det = tab[nm]
                if vlsr_thresh is None:
                    vOK = tab['host' + nm[-1] + '_vdevok']
                else:
                    vOK = tab['host{}_vlsr{}ok'.format(nm[-1], vlsr_thresh.to(u.km/u.s).value)]

                dHVCok = tab['host' + nm[-1] + '_dHVCok']
                tab['findable' + suffix] = det & vOK & dHVCok


def real_lg_in_galfa_dr2(lgtab, hicut=True):
    """
    `lgtab` is from the mcconachie catalog
    """
    ingalfa = lgtab[(lgtab['decdeg']>0)&(lgtab['decdeg']<40)&(lgtab['vh']<600)]
    if hicut:
        ingalfa = ingalfa[(ingalfa['MHI']>0)&(ingalfa['MHI']<99)]
    return ingalfa
