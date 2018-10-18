#!/usr/bin/env bash

echo "building quantizer extension module..."
python setup.py build_ext --inplace
