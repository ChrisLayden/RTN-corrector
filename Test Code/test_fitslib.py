import numpy as np
from astropy.io import fits
import sys, os
mydir = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(mydir, '..')))
from fitsreader import FITSReader

files = ('../Sample Data/bias_data/bias_cutouts_cube.fits',)
correct = fits.getdata(files[0])
psr = FITSReader(files)

for i, row in psr:
    if not np.array_equal(row, correct[:, i:i+1, :]):
        print('FitsPixelReader failed')
        print('row', row.shape)
        print('correct', correct[:, i:i+1, :].shape)
        print(correct[:, i:i+1, :10])
        break
else:
    print('FITSReader row generation ok.')
