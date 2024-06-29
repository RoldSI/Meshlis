import ctype

# Load the shared library
lib = ctypes.CDLL('./mic/target/debug/libserver.dylib')
# Define the argument and return types of the function
lib.add.argtypes = (ctypes.c_int, ctypes.c_int)
lib.add.restype = ctypes.c_int


