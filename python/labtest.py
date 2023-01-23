import processing as pr
import localization as loc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

PEAKSTODETECT = 20
INTERVALSIZE = 0.5
SOFSOUND_WATER = 1430.3
INITSTEPSIZE = 0.99
MINIMUMTHRESH = 0.1

def main():
    fs = 200e3
    _, tSigSum, lstAnchors = pr.simulation([5e-3, 10e-3], 2, showAll=False, addGWN=False, useSim=False, polyDeg=10, labExport=True)
    #pr.simulation([5e-3, 10e-3], 2, showAll=False, addGWN=False, useSim=False, polyDeg=9, labExport=True)
    #pr.simulation([5e-3, 10e-3], 2, showAll=False, addGWN=False, useSim=False, polyDeg=8, labExport=True)
    # use filt (not filtfilt) to fitler the signal by fitler b and hydrophone vals
    # gated periods
    # 1.5V

    tSigBBrSum = tSigSum[1][:int(PEAKSTODETECT * INTERVALSIZE * fs)]
    
    tSigBBrSum = np.pad(tSigBBrSum, (int(INTERVALSIZE * fs), 0), 'mean')
    tSum = tSigSum[0][:int(PEAKSTODETECT * INTERVALSIZE * fs)]

    lstCFARSig = []
    anchorPeaks = []
    #anchorPeaksSnd = []
    guLenInSec = 0.00014
    guLen = int(fs * guLenInSec)
    for i in range(len(lstAnchors)):
        peaks = []

        anchor = np.append(lstAnchors[i], [0] * (len(tSigBBrSum) - len(lstAnchors[i])))
        tauCC, tauSigCC = pr.corr_lag(tSigBBrSum, anchor, fs) 

        far = 200.0e-1
        groups = []

        prev_len = None
        stepsize = INITSTEPSIZE

        print("num of pk areas,", "stepsize,", "far")
        while len(groups) != PEAKSTODETECT:
            
            # 2.1e-1
            cfarSigSum = pr.ca_cfar(tauSigCC, guLen * 6, guLen, far, sort=False, threshold=MINIMUMTHRESH)

            

            #altPeaksInd, prop = find_peaks(np.abs(tauSigCC), height=0.2, distance=int(fs*0.49))
            #altPeaksInd = altPeaksInd[np.argsort(-prop['peak_heights'])][:PEAKSTODETECT]

            sigCCpks = np.abs(tauSigCC.copy())
            sigCCInd = np.where(sigCCpks > cfarSigSum)

            # group consecutive values for further processing
            groups = loc.group_consecutive_values(sigCCInd[0], int(INTERVALSIZE * 0.95 * fs))

            #if len(groups) != PEAKSTODETECT:
            #    print(len(groups))
            #    print("Peaks not succefully detected!")
            #if len(groups) > PEAKSTODETECT:
            far *= stepsize
            #else:
            #   far /= stepsize
            
            #if prev_len is not None:
            #    if prev_len == len(groups):
            #        stepsize -= 0.01
            #    else:
            #        stepsize = INITSTEPSIZE

            prev_len = len(groups)
            print(len(groups), stepsize, far)

        lstCFARSig.append((tauCC, tauSigCC, cfarSigSum))

        for gr in groups:
            # get maximum of every index group
            cfarPeakInd = np.where(np.abs(tauSigCC) == np.max(np.abs(tauSigCC[gr])))[0][0]
            #peaks.append(tauCC[cfarPeakInd])
            peaks.append(cfarPeakInd)
        
        anchorPeaks.append(peaks)
        #anchorPeaksSnd.append(altPeaksInd)

    tdoaLineplot = make_subplots(rows=2)

    tauCC = lstCFARSig[0][0]
    tauSigCC = lstCFARSig[0][1]
    cfarSigSum = lstCFARSig[0][2]

    tdoaLineplot.add_trace(go.Scatter(x=tauCC, y=np.abs(tauSigCC), mode='lines', marker_color='#000'), row=1, col=1)
    tdoaLineplot.add_trace(go.Scatter(x=tauCC, y=cfarSigSum, mode='lines', marker_color='#636EFA'), row=1, col=1)
    for peak in anchorPeaks[0]:
        tdoaLineplot.add_vline(tauCC[peak], line_color='#00CC96', line_width=3, line_dash='dash', row=1, col=1)
    #for peak in anchorPeaksSnd[0]:
    #    tdoaLineplot.add_vline(tauCC[peak], line_color='#00CC96', line_width=3, line_dash='dash', row=1, col=1)



    tauCC = lstCFARSig[1][0]
    tauSigCC = lstCFARSig[1][1]
    cfarSigSum = lstCFARSig[1][2]

    tdoaLineplot.add_trace(go.Scatter(x=tauCC, y=np.abs(tauSigCC), mode='lines', marker_color='#000'), row=2, col=1)
    tdoaLineplot.add_trace(go.Scatter(x=tauCC, y=cfarSigSum, mode='lines', marker_color='#636EFA'), row=2, col=1)
    for peak in anchorPeaks[1]:
        tdoaLineplot.add_vline(tauCC[peak], line_color='#00CC96', line_width=3, line_dash='dash', row=2, col=1)
    #for peak in anchorPeaksSnd[1]:
    #    tdoaLineplot.add_vline(tauCC[peak], line_color='#00CC96', line_width=3, line_dash='dash', row=2, col=1)


    tdoaLineplot.update_layout(showlegend=False, title="CFAR TOA correlation peaks")
    


    fstAnchTOA = np.divide(anchorPeaks[0], fs)
    sndAnchTOA = np.divide(anchorPeaks[1], fs)


    lstTDOA = []
    for i in range(len(fstAnchTOA)):
        lstTDOA.append(round(np.abs(fstAnchTOA[i] - sndAnchTOA[i]), 6))

    print("TDOAs in s", lstTDOA)

    lstDist = np.multiply(lstTDOA, SOFSOUND_WATER)

    dstPlot = go.Figure(go.Scatter(x=tuple(range(PEAKSTODETECT)), y=lstDist))
    dstPlot.add_hline(4.1, name="measured distance")
    dstPlot.update_layout(title="Approximated vs Measured Distances in m")
    dstPlot.show()

    print("distances in m", lstDist)
    print("mean dist in m", np.mean(lstDist))

    tdoaLineplot.show()


if __name__ == "__main__":
    main()
    