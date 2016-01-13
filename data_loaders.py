import os
import re
from glob import glob

from astropy import units as u
from astropy.table import Table, QTable

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
def load_galfa_sensitivity(fn):
    raise NotImplementedError
