from itertools import combinations
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sequences import gold_seq
from polys import GOLD

from commpy import filters as flt
from scipy import signal as sig, ndimage as img
from scipy.io import savemat
from math import comb

MATPATH = '/Users/sk/Library/CloudStorage/OneDrive-Persönlich/Studium/TUHH/3. Semester Master/Forschungsprojekt/keller_orthogonal-codes/python/output/signal.mat'

### signal parameters
fs = 200e3          # sampling rate
bw = 20e3           # bandwith
Tsym = 1 / bw       # symbol length
SpS = fs * Tsym     # upsampling factor
rolloff = 1/8       # FIR cosine filter coefficent
fc = 62.5e3         # carrier freqency

### gold sequence generation
deg = 10
codeLen = 2 ** deg + 1

def _get_fftfunc(sig: np.ndarray, fs: float):

    fSig = np.fft.fft(sig)
    freq = np.fft.fftfreq(fSig.shape[0], 1/fs)
    fSig = np.fft.fftshift(fSig)
    freq = np.fft.fftshift(freq)

    return (freq, np.abs(fSig))

def gen_TB_signal(code: np.ndarray):

    ### upsample and filter via FIR cosine
    filter = flt.rcosfilter(1024, rolloff, Tsym, fs)[1]
    tSigBB = sig.resample_poly(code, SpS, 1, window=filter)
    time = np.linspace(0, Tsym * codeLen, len(tSigBB))

    ### shift spectrum to transmission band
    tSigTB = np.real(tSigBB * np.exp(2.j*np.pi*fc*time))

    matobj = {'tSigTB': tSigTB, 'time': time}
    savemat(MATPATH, matobj)

    return (time, tSigTB)

def get_BB_signal(time: np.ndarray, tSigTB: np.ndarray):

    ### reshift spectrum to baseband
    tSigBBr = np.real(tSigTB * np.exp(2.j*np.pi*-fc*time))

    ### applying 3rd order butterworth low pass forward and backward
    b, a = sig.butter(3, bw, 'lowpass', fs=fs, output='ba')
    tSigBBr = sig.filtfilt(b, a, tSigBBr)

    return (time, tSigBBr)

def delay_sum(signals: list[np.ndarray], delays: list[float], fs: float):
    singalSumLen = int(np.floor(np.max(delays) * fs) + len(signals[0]) + 1)
    signalSum = np.zeros(singalSumLen)
    timeSum = np.linspace(0, Tsym * codeLen, singalSumLen)
    
    for s, d in zip(signals, delays):
        index = range(int(np.floor(d * fs) + 1), int((np.floor(d * fs) + len(s) + 1)))
        signalSum[index] = signalSum[index] + s

    return (timeSum, signalSum)

def corr_lag(x : np.ndarray, y: np.ndarray, fs: float):
    sigLen = len(x)
    tCC = sig.correlate(x, y, 'same')
    normDiv = np.sqrt(sig.correlate(x, x, 'same')[int(sigLen/2)] * sig.correlate(y, y, 'same')[int(sigLen/2)])
    tCC /= normDiv
    tLags = np.linspace(-0.5 * sigLen/fs, 0.5 * sigLen/fs, sigLen)
    tCC = tCC[tLags > 0]
    tLags = tLags[tLags > 0]

    return (tLags, tCC)

def ca_cfar(x: np.ndarray, trBinSize: int, guBinSize: int, faRate: float):


    # +-----------+-----------+-----------+-----------+-----------+
    # | train bin | guard bin | candidate | guard bin | train bin |
    # +-----------+-----------+-----------+-----------+-----------+


    binSize = x.size
    sideSize = trBinSize + guBinSize


    alpha = trBinSize * 2 * (faRate ** (-1/(trBinSize * 2)) - 1)

    threshList = []


    for i in range(sideSize, binSize - sideSize):
        
        fstSum = np.sum(x[i-sideSize:i+sideSize+1])
        sndSum = np.sum(x[i-guBinSize:i+guBinSize+1])

        estNoise = (fstSum - sndSum) / (trBinSize * 2)
        threshList.append(alpha * estNoise)

    return np.pad(threshList, (sideSize, binSize - sideSize), 'edge')



def main():

    startSeed = 32

    numOfAnchors = 5

    figure = make_subplots(rows=numOfAnchors, cols=1)
    tSendSig = []

    tDelays = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]

    for i in range(startSeed, startSeed + numOfAnchors):
        rawCode = gold_seq(deg, i, 2)[0]
        signal = gen_TB_signal(rawCode)
        signal = get_BB_signal(signal[0], signal[1])
        tSendSig.append(signal[1])
        figure.add_trace(go.Scatter(x=signal[0], y=signal[1], mode='lines', marker_color='#000'), row=i-(startSeed-1), col=1)

    figure.show() 

    numOfCorr = comb(numOfAnchors, 2)

    figure = make_subplots(rows=numOfCorr, cols=2)

    index = 1
    for pair in combinations(tSendSig, 2):
        SigCC = sig.correlate(pair[0], pair[1], 'same')
        tauCC = sig.correlation_lags(len(pair[0]), len(pair[1]), 'same')
        figure.add_trace(go.Scatter(x=tauCC, y=SigCC, mode='lines', marker_color='#000'), row=index, col=1)
        index += 1

    index = 1
    for si in tSendSig:
        SigAC = sig.correlate(si, si, 'same')
        tauAC = sig.correlation_lags(len(si), len(si), 'same')
        figure.add_trace(go.Scatter(x=tauAC, y=SigAC, mode='lines', marker_color='#000'), row=index, col=2)
        index += 1
    figure.update_layout(showlegend=False)
    figure.show() 

    figure = make_subplots(rows=1, cols=1)
    tSigSum = delay_sum(tSendSig, tDelays, fs)
    figure.add_trace(go.Scatter(x=tSigSum[0], y=tSigSum[1], mode='lines', marker_color='#000'), row=1, col=1)
    figure.update_layout(showlegend=False)
    figure.show() 

    figure = make_subplots(rows=numOfAnchors, cols=1)
    index = 1
    for si in tSendSig:
        si = np.append(si, [0] * (len(tSigSum[1]) - len(si)))
        SigCC = corr_lag(tSigSum[1], si, fs)
        figure.add_trace(go.Scatter(x=SigCC[0], y=SigCC[1], mode='lines', marker_color='#000'), row=index, col=1)
        lagInd = np.argmax(SigCC[1])
        figure.add_vline(SigCC[0][lagInd], line_color='#EF553B', line_width=3, line_dash='dash', row=index, col=1)
        index += 1
    figure.update_layout(showlegend=False)
    figure.show() 

def slider():
    startSeed = 32

    numOfAnchors = 3

    figure = make_subplots(rows=numOfAnchors, cols=1)
    tSendSig = []

    tDelays = [4e-3, 5e-3, 10e-3]
    for i in range(startSeed, startSeed + numOfAnchors):
        rawCode = gold_seq(deg, i, 2)[0]
        signal = gen_TB_signal(rawCode)
        signal = get_BB_signal(signal[0], signal[1])
        tSendSig.append(signal[1])

    tSigSum = delay_sum(tSendSig, tDelays, fs)
    #gwn = np.random.normal(0, 19, len(tSigSum[1]))
    #tSigSum[1][:] += gwn

    winLen = fs * 2e-4
    alpha = 50

    figure = make_subplots(rows=numOfAnchors, cols=1)
    index = 1
    for si in tSendSig:

        si = np.append(si, [0] * (len(tSigSum[1]) - len(si)))
        SigCC = corr_lag(tSigSum[1], si, fs)
        figure.add_trace(go.Scatter(x=SigCC[0], y=SigCC[1], mode='lines', marker_color='#000'), row=index, col=1)

        # applying estimate for upprt threshold
        
        #varSigSum = alpha * (wins.std(axis=-1) ** 2)
        #varSigSum = np.append(varSigSum, [0] * (len(SigCC[1]) - len(varSigSum)))
        #figure.add_trace(go.Scatter(x=SigCC[0], y=varSigSum, mode='lines', marker_color='#636EFA'), row=index, col=1)
        varSigSum = []

        varSigSum = np.abs(ca_cfar(SigCC[1], int(winLen), int(0.1 * winLen), 1e-3))

        #wins = np.lib.stride_tricks.sliding_window_view(varSigSum, 60)
        #varSigSum = np.ma.average(wins, axis=-1)
        
        figure.add_trace(go.Scatter(x=SigCC[0], y=varSigSum, mode='lines', marker_color='#636EFA'), row=index, col=1)
        lagInd = np.argmax(SigCC[1])
        figure.add_vline(SigCC[0][lagInd], line_color='#EF553B', line_width=3, line_dash='dash', row=index, col=1)
        index += 1
    figure.update_layout(showlegend=False)
    figure.show() 



if __name__ == "__main__":
    #main()
    slider()
