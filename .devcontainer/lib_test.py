import os
import numpy

print("NumPy file:", numpy.__file__)

try:
    numpy_core = numpy.core._multiarray_umath.__file__
    print("NumPy core:", numpy_core)
    
    if os.path.exists(numpy_core):
        print("Shared library dependencies:")
        os.system(f"ldd {numpy_core}")
    else:
        print("NumPy core file not found.")
except AttributeError:
    print("NumPy core module not found.")