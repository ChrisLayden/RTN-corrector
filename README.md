# RTN-corrector
Code for removing random telegraph noise (RTN) from astronomical images.

Use rtn_fitter.py to identify RTN in pixels of your detector and parametrize this noise in terms of a sum of three Gaussians. This requires a stack of at least 100 bias frames. This script will output a fits file containing the parametrizations for all pixels with RTN.

Use rtn_fixer.py to apply RTN correction to a stack of astronomical images. This script takes a rolling average of every N (default 10) frames and identifies any events in which a pixel jumped relative to this average. If the jump is consistent with RTN, the pixel signal is corrected. A fits file with RTN parameters must exist to run this script.
