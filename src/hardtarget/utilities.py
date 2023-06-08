# Stuff that is useful in several places in hardtarget library

# Juha's convenience function for reading 1d vector of c8 values from a digital_rf source
def read_vector_c81d(rdf_reader, *args, **kws):
    return rdf_reader.read_vector_1d(*args, **kws).astype('c8', casting='unsafe', copy=False)


# From Stuffr
# import datetime
# import numpy
# def unix2date(x):
#     return datetime.datetime.utcfromtimestamp(x)

# def unix2datestr(x):
#     d0=numpy.floor(x/60.0)
#     frac=x-d0*60.0
#     stri=unix2date(x).strftime('%Y-%m-%d %H:%M:')
#     return("%s%02.2f"%(stri,frac))

# def sec2dirname(t):
#     return(unix2date(t).strftime("%Y-%m-%dT%H-00-00"))