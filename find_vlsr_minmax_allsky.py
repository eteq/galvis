import numpy as np

def find_vlsr_minmax(l, b):
    b = b/180.*np.pi
    l = l/180.*np.pi

    R0 = 8.5 # kpc
    nlos = 100  # data points along the line of sight
    d = np.linspace(0, 50, nlos)
    Rlos = R0*np.sqrt((d/R0*np.cos(b))**2 - 2*np.cos(b)*np.cos(l)*(d/R0) + 1) # Galactocentric R for a given d along los
    zlos = d*np.sin(b)    # z height for a given d along los

    z1 = 1
    z2 = 3
    zmax = np.zeros(Rlos.size) + z1              # the disk has a thickness of 2 kpc at gr<R0
    zmax[Rlos>=R0] = z1+(z2-z1)*((Rlos[Rlos>=R0]/R0-1)**2)/4.  # disk thickness increase to 6 kpc at gr=3R0

    # Rotation cuve / rigid body
    vR = np.zeros(Rlos.size)+220
    vR[Rlos<0.5] = Rlos[Rlos<0.5]/R0*220

    # VLSR for a given d along los
    vlsr = (R0/Rlos*vR - 220)*np.sin(l)*np.cos(b)

    Rmax = 26 # kpc
    ii_dmax = np.fabs(zlos)<=zmax
    jj_Rmax = Rlos<=Rmax
    ind = np.mgrid[0:nlos:1]
    if ind[ii_dmax].size < ind[jj_Rmax].size:
        vlsr_minmax = vlsr[ind[ii_dmax]].min(), vlsr[ind[ii_dmax]].max()
    else:
        vlsr_minmax = vlsr[ind[jj_Rmax]].min(), vlsr[ind[jj_Rmax]].max()
    return vlsr_minmax

def find_vlsr_minmax_allsky():
    b = np.mgrid[-90:90.1:0.5]
    l = np.mgrid[0:360:0.5][::-1]
    vlsr_min_allsky = np.zeros((b.size, l.size))
    vlsr_max_allsky = np.zeros((b.size, l.size))
    for ib in range(b.size):
        for il in range(l.size):
            vmin, vmax = find_vlsr_minmax(l[il], b[ib])
            vlsr_min_allsky[ib, il] = vmin
            vlsr_max_allsky[ib, il] = vmax
    return b, l, vlsr_min_allsky, vlsr_max_allsky
