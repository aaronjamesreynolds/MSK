#!/usr/bin/env python

import yaml

"""Reads in nuclear data from a yaml input file"""


class Input:

    def __init__(self, input_file_name):

        with open(input_file_name, 'r') as stream:
            try:
                self.data = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)


class InputDataYAML(yaml.YAMLObject):

    yaml_tag = u'!InputData'

    def __init__(self, sig_t, sig_s, sig_f, nu, cells, material, dx, dt):
        self.sig_t = sig_t
        self.sig_s = sig_s
        self.sig_f = sig_f
        self.nu = nu
        self.cells = cells
        self.material = material
        self.dx = dx
        self.dt = dt

    def __repr__(self):
        return "%s(sig_t = %r, sig_s=%r, sig_f=%r, nu=%r, cells = %r, material = %r, dx = %r, dt = %r)" % (self.__class__.__name__,
                                                                                         self.sig_t, self.sig_s,
                                                                                         self.sig_f, self.nu,
                                                                                         self.cells, self.material,
                                                                                         self.dx, self.dt)

# if __name__ == '__main__':
#
#     test = Input("test_input.yaml")
#     print test.data