#!/usr/bin/env python

import input.read as read
import os


def test_initialize_input_instance():

    current_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_directory, 'testing_files/test_input.yaml')
    nuclear_data = read.Input(file_path)
