import numpy as np
import plotly.graph_objects as go
from commpy import filters as flt
from plotly.subplots import make_subplots
from scipy import signal as sig
from scipy.io import savemat, loadmat, wavfile
from sequences import gold_seq

MATSAVE = '/Users/sk/Library/CloudStorage/OneDrive-Persönlich/Studium/TUHH/3. Semester Master/Forschungsprojekt/uw-watermark-main/Watermark/input/signals/sigTB_'
WAVLOAD = '/Users/sk/Library/CloudStorage/OneDrive-Persönlich/Studium/TUHH/3. Semester Master/Forschungsprojekt/uw-watermark-main/Watermark/output/'
CHANNEL = 'PLANE1'


file_index = 0
load_index = 0

### signal parameters
fs = 200e3          # sampling rate
bw = 20e3           # bandwith
Tsym = 1 / bw       # symbol length
SpS = fs * Tsym     # upsampling factor
rolloff = 1/8       # FIR cosine filter coefficent
fc = 62.5e3         # carrier freqency
sysOrd = 5          # order of butterworth filter

watermarkDelay = 12e-3

### gold sequence generation
deg = 10
codeLen = 2 ** deg + 1

### time axis
#time = np.linspace(0, Tsym * codeLen, len(tSigBB))

def _get_fftfunc(sig: np.ndarray, fs: float):

    fSig = np.fft.fft(sig)
    freq = np.fft.fftfreq(fSig.shape[0], 1/fs)
    fSig = np.fft.fftshift(fSig)
    freq = np.fft.fftshift(freq)

    return (freq, np.abs(fSig))

def gen_TB_signal(time: np.ndarray, tSigBB: np.ndarray, createMat: bool = False):

    global file_index

    ### shift spectrum to transmission band
    tSigTB = np.real(tSigBB * np.exp(2.j*np.pi*fc*time))

    if createMat:
        ### save signal as matlab object for watermark
        matobj = {'fs_x': fs, 'nBits': 0, 'x': tSigTB}
        savemat(MATSAVE + str(file_index) + '.mat', matobj)
        file_index += 1

    return (time, tSigTB)

def get_BB_signal(time: np.ndarray, tSigTB: np.ndarray, loadMat: bool = False):

    global load_index

    filepath = CHANNEL + "/sigTB_" + str(load_index) + "/" + CHANNEL + "_001.wav"

    sigLen = len(tSigTB)
    if loadMat:
        x, tSigTB = wavfile.read(WAVLOAD + filepath)
        #tSigTB = tSigTB[:10230]
        tSigTB = tSigTB[int(watermarkDelay * fs):(sigLen + int(watermarkDelay * fs))]
        print(load_index)
        load_index += 1

    ### reshift spectrum to baseband
    tSigBBr = tSigTB * np.exp(2.j*np.pi*-fc*time)

    ### applying 3rd order butterworth low pass forward and backward
    b, a = sig.butter(sysOrd, bw, 'lowpass', fs=fs, output='ba')
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

    # ┌-----------┬-----------┬-----------┬-----------┬-----------┐
    # | train bin | guard bin | candidate | guard bin | train bin |
    # └-----------┴-----------┴-----------┴-----------┴-----------┘

    x = np.abs(x)
    sigLen = x.size
    binSize = trBinSize + guBinSize

    alpha = trBinSize * 2 * (faRate ** (-1/(trBinSize * 2)) - 1)

    threshList = []

    for i in range(binSize, sigLen - binSize):

        binSum = np.sum(x[i-binSize:i+binSize+1])
        guBinSum = np.sum(x[i-guBinSize:i+guBinSize+1])

        estThresh = (binSum - guBinSum) / (trBinSize * 2)
        estThresh *= alpha
        threshList.append(estThresh)

    return np.pad(threshList, (binSize, sigLen - binSize), 'edge')

def showButterBode():
       
    sys = sig.butter(sysOrd, bw, 'lowpass', fs=fs)
    fBode, dBMag, fPha = sig.bode(sys)

    figure = make_subplots(rows=2, cols=1, x_title='Frequency [deg]', subplot_titles=('Magnitude [dB]', 'Phase [deg]'))
    figure.add_trace(go.Scatter(x=fBode, y=dBMag, fill='tozerox', marker_color='#EF553B'), row=1, col=1)
    figure.add_trace(go.Scatter(x=fBode, y=fPha, fill='tozerox'), row=2, col=1)
    figure.update_xaxes(type='log')
    figure.update_layout( title='Butterworth Low-Pass of Order 5 and 20kHz cutoff', showlegend=False)
    figure.show()

def get_snr_noise(signal: np.ndarray, snr: float):

    stdEst = np.sqrt(1/(snr * np.mean(signal ** 2)))
    gwn = np.random.normal(0, stdEst, len(signal))

    return gwn


def simulator(tDelays: list[float], numOfAnchors: int, addGWN = False, startSeed = 42, showAll = False):

    figure = make_subplots(rows=numOfAnchors, cols=1)
    tSendSig = []
    
    for i in range(startSeed, startSeed + numOfAnchors):

        ### generate gold code
        rawCode = gold_seq(deg, i, 2)[0]

        ### upsample and filter via FIR cosine
        filter = flt.rcosfilter(1024, rolloff, Tsym, fs)[1]
        tSigBB = sig.resample_poly(rawCode, SpS, 1, window=filter)
        time = np.linspace(0, Tsym * codeLen, len(tSigBB))

        print("sigBB len", len(tSigBB))

        ### shift spectrum to transmission band
        _, tSigTB = gen_TB_signal(time, tSigBB, True)

        ### add white noise (@TODO via SNR)
        if addGWN:
            gwn = get_snr_noise(tSigTB, 1000)
            tSigTB[:] += gwn

        ### shift back to baseband
        _, tSigBBr = get_BB_signal(time, tSigTB, True)

        tSendSig.append(tSigBBr)

    tSigSum = delay_sum(np.real(tSendSig), tDelays, fs)
    winLen = fs * 5e-4

    figure = make_subplots(rows=numOfAnchors, cols=1)
    index = 1
    for si in tSendSig:

        si = np.append(si, [0] * (len(tSigSum[1]) - len(si)))
        SigCC = corr_lag(np.real(tSigSum[1]), np.real(si), fs)
        varSigSum = ca_cfar(SigCC[1], int(winLen), int(0.1 * winLen), 1e-3)
        lagInd = np.argmax(SigCC[1])

        figure.add_trace(go.Scatter(x=SigCC[0], y=SigCC[1], mode='lines', marker_color='#000'), row=index, col=1)
        figure.add_trace(go.Scatter(x=SigCC[0], y=varSigSum, mode='lines', marker_color='#636EFA'), row=index, col=1)
        figure.add_vline(SigCC[0][lagInd], line_color='#EF553B', line_width=3, line_dash='dash', row=index, col=1)

        index += 1

    figure.update_layout(showlegend=False)
    figure.show() 

    if showAll:

        fBB, fSigBB = _get_fftfunc(tSigBB, fs)
        fTB, fSigTB = _get_fftfunc(tSigTB, fs)
        fBBr, fSigBBr = _get_fftfunc(tSigBBr, fs)

        ### code to transfer band process plots and backwards process plots

        figure = make_subplots(rows=3, cols=1)
        figure.add_trace(go.Bar(y=rawCode, name="bit domain", marker_color='#19D3F3'), row=1, col=1)
        figure.add_trace(go.Scatter(x=time, y=tSigBB, mode='lines', name="baseband signal", marker_color='#EF553B'), row=2, col=1)
        figure.add_trace(go.Scatter(x=time, y=tSigTB, mode='lines', name="carrier signal", marker_color='#636EFA'), row=2, col=1)
        figure.add_trace(go.Scatter(x=time, y=np.real(tSigBBr), mode='lines', name="baseband signal", marker_color='#FFA51A'), row=2, col=1)
        figure.add_trace(go.Scatter(x=fBB, y=np.abs(fSigBB), fill='tozeroy', mode='lines', name="baseband", marker_color='#EF553B'), row=3, col=1)
        figure.add_trace(go.Scatter(x=fTB, y=np.abs(fSigTB), fill='tozeroy', mode='lines', name="carrier", marker_color='#636EFA'), row=3, col=1)
        figure.add_trace(go.Scatter(x=fBBr, y=np.abs(fSigBBr), fill='tozeroy', mode='lines', name="recv baseband", marker_color='#FFA51A'), row=3, col=1)
        figure.update_layout(title='Baseband & Transmission-Band Processing')
        figure.show()






def main():
    simulator([2e-3, 4e-3, 8e-3], 3, showAll=False, addGWN=False)
    #showButterBode()

if __name__ == "__main__":
    main()
    






"""def main():

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
    figure.show() """