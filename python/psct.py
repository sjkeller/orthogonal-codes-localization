import numpy as np
import matplotlib.pyplot as plt
from pip import main

def genPRBS(cinit: int, length: int, signed: bool = True):

    # intitialize random generator and get starting value
    np.random.seed(cinit)
    seq = np.zeros(length)
    code = np.random.rand(1)

    for i in range(length):
        code = np.round(code)
        code = code.astype(int)

        # generate PRBS-31 with LFSR
        next_code = (~((code<<1)^(code<<4)) & 0xFFFFFFF0)
        next_code |= (~(( (code<<1 & 0x0E) | (next_code>>31 & 0x01)) ^ (next_code>>28)) & 0x0000000F)
        code = next_code
        seq[i] = code

    # encode random signal to random binary or signed sequence
    smean = np.mean(seq)
    if signed:
        seq = np.where(seq >= smean, 1, -1)
    else:
        seq = np.where(seq >= smean, 1, 0)

    return seq


def genBaseband(cinit: int, M: int, T: int, fs: int, rolloff: int):
    return


seq = genPRBS(13, 300)
print(seq)

plt.figure() 
plt.plot(seq, drawstyle='steps', label='PRBS-31')
plt.legend()
plt.show()

def main():
    x = 
    code = [int(d) for d in str(oct(x))[2:]]

if __name__ == "__main__":
    main()