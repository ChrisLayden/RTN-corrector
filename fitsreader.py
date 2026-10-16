# Copyright (c) 2026 David Whysong
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
from astropy.io import fits


class FITSReader:

    def __init__(self, filelist, ext=0):
        self.files = tuple(f for f in filelist)
        self.ext = ext
        self.fits = None
        with fits.open(self.files[0]) as ff:
            self.nx, self.ny = ff[ext].shape[-2:]

    @property
    def depth(self):
        # Assumes the data shape is either 2 or 3-dimensional
        return tuple(1 if len(f.shape) == 2 else f.shape[0] for f in self.fits)


    def _get_rows(self, findex, start, end):
        ndims = len(self.fits[findex].shape)
        if ndims == 2:
            _data = self.fits[findex].section[start:end, :]
            return _data[np.newaxis, :, :]
        elif ndims == 3:
            return self.fits[findex].section[:, start:end, :]
        else:
            raise Exception

    def get_rows(self, start, end, frames_to_keep=None):
        self.open()
        depths = self.depth
        if frames_to_keep is None:
            frames_to_keep = sum(self.depth)

        results = []
        depth = 0
        for findex in range(len(self.fits)):
            results.append(self._get_rows(findex, start, end))
            depth += results[-1].shape[0]
            if depth > frames_to_keep:
                break
        self.close()
        result = np.concatenate(results)
        return result[:frames_to_keep, :, :]

    def open(self):
        if self.fits is None:
            self._handles = tuple(fits.open(path) for path in self.files)
            self.fits = tuple(h[self.ext] for h in self._handles)

    def close(self):
        if self.fits is not None:
            for f in self._handles:
                f.close()
            # Explicitly delete to force freeing of resources
            del self._handles
            del self.fits
            self.fits = None
