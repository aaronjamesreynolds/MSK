#!/usr/bin/env python

import yaml

"""Reads in nuclear data from a yaml input file"""

class InputData(yaml.YAMLObject):
    yaml_tag = u'!InputData'

    def __init__(self, sig_t, sig_s, sig_f, nu, cells, material):
        self.sig_t = sig_t
        self.sig_s = sig_s
        self.sig_f = sig_f
        self.nu = nu
        self.cells = cells
        self.material = material

    def __repr__(self):
        return "%s(sig_t = %r, sig_s=%r, sig_f=%r, nu=%r, cells = %r, material = %r)" % (self.__class__.__name__,
                                                                                         self.sig_t, self.sig_s,                                                                                     self.sig_f, self.nu,
                                                                                         self.cells, self.material)
if __name__ == '__main__':
