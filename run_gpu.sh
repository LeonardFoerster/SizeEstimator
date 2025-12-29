#!/bin/bash
# Wrapper script to run size_estimator.py on AMD RX 7000 series (RDNA3)
# Fixes "Illegal seek for GPU arch : gfx1101" error by masquerading as gfx1100

export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Pass all arguments to the python script
se/bin/python size_estimator.py "$@"
