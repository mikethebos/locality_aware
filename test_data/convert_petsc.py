import scipy.io, PetscBinaryIO

def convert(file_in, file_out):
    A = scipy.io.mmread(file_in)
    A = A.tocsr()
    PetscBinaryIO.PetscBinaryIO().writeMatSciPy(open(file_out,'w'), A)

if __name__ == "__main__":
    import sys
    fn_in = sys.argv[1]
    fn_out = fn_in.removesuffix(".mtx") + ".pm"
    convert(fn_in, fn_out)