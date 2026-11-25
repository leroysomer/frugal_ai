import os, sys
import matplotlib.pyplot as plt
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np


l_size = 512

# if os.system("gcc matmul.c -o matmul -lm -lopenblas"): sys.exit("Compilation error")
if os.system("gcc matmul.c -o matmul -O3 -ffast-math "
             "-I/opt/homebrew/opt/openblas/include "
             "-L/opt/homebrew/opt/openblas/lib -lopenblas -lm"):
    
    sys.exit("Compilation error")
c_code_output = os.popen("./matmul %d"%(l_size)).read()

if "ERROR ! MATRIX DIFF !" in c_code_output:
    print(c_code_output)
    exit()

openblas_time, custom_time = [float(line.split()[-2]) for line in c_code_output.splitlines()]

print ("OpenBLAS at size %d \t time %f s"%(l_size, openblas_time))
print ("Custom at size %d \t time %f s"%(l_size, custom_time))

#Use this wrapper to create loops over l_size for the different versions of matmul in C and plot the FLOPS curves

def curves(start, stop, step):
	
    sizes = np.arange(start, stop, step)
    times = np.zeros((2, len(sizes)))

    for i, size in enumerate(sizes):
        c_code_output = os.popen("./matmul %d"%(size)).read()
        openblas_time, custom_time = [float(line.split()[-2]) for line in c_code_output.splitlines()]
        times[0][i] = openblas_time
        times[1][i] = custom_time

    print(times[0])
    flops = sizes**3/times

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, flops[0], label="openblas")
    ax.plot(sizes, flops[1], label="custom")
    ax.legend()
    ax.set_title('Numerical efficiency')
    ax.set_xlabel('Matrix size')
    ax.set_ylabel('GFLOPS')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.show()
    return

curves(128, 1920, 128)
