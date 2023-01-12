from processing import simulation, corr_lag, ca_cfar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
# source pending
SOFSOUND_WATER = 1500
POSITIONS = 15

def group_consecutive_values(input_list):
    result = []
    current_group = [input_list[0]]
    for i in range(1, len(input_list)):
        if input_list[i] == input_list[i-1] + 1:
            current_group.append(input_list[i])
        else:
            result.append(current_group)
            current_group = [input_list[i]]
    result.append(current_group)
    return result

def totalError(listRealPos: list[tuple], listEstPos: list[tuple]):
    errors = []
    for i in range(len(listRealPos)):
        realPos = np.array(listRealPos[i])
        estPos = np.array(listEstPos[i])
        errors.append(np.linalg.norm(estPos - realPos))

    return errors

def quadraticApprox(anchorPos: list[tuple], toas: list[float], lastZPos: float = 0.0):
    """ TDOA Alogrithm for localaization by four anchors 

    Args
    ----
    anchorPos: list[tuple]
        anchors positions in 3d space
    toas: list[float]
        time of arrivals in seconds
    zBound: float
        lower threshold for z candidate

    Returns
    -------
    position of target in 3d space
    """

    d01 = SOFSOUND_WATER * (toas[0] - toas[1])
    d02 = SOFSOUND_WATER * (toas[0] - toas[2])
    d21 = SOFSOUND_WATER * (toas[2] - toas[1])
    d23 = SOFSOUND_WATER * (toas[2] - toas[3])

    print("distances:", d01, d02, d21, d23)

    x0, y0, z0 = anchorPos[0]
    x1, y1, z1 = anchorPos[1]
    x2, y2, z2 = anchorPos[2]
    x3, y3, z3 = anchorPos[3]

    x10 = x1 - x0
    x20 = x2 - x0
    x12 = x1 - x2
    x32 = x3 - x2

    y10 = y1 - y0
    y20 = y2 - y0
    y12 = y1 - y2
    y32 = y3 - y2

    z10 = z1 - z0
    z20 = z2 - z0
    z12 = z1 - z2
    z32 = z3 - z2

    varA = (d02 * x10 - d01 * x20) / (d01 * y20 - d02 * y10)
    varB = (d02 * z10 - d01 * z20) / (d01 * y20 - d02 * y10)
    varC = (d02 * (d01 ** 2 + x0 ** 2 - x1 ** 2 + y0 ** 2 - y1 ** 2 + z0 ** 2 - z1 ** 2) - d01 * (d02 ** 2 + x0 ** 2 - x2 ** 2 + y0 ** 2 - y2 ** 2 + z0 ** 2 - z2 ** 2)) / (2 * (d01 * y20 - d02 * y10))
    
    varD = (d23 * x12 - d21 * x32) / (d21 * y32 - d23 * y12)
    varE = (d23 * z12 - d21 * z32) / (d21 * y32 - d23 * y12)
    varF = (d23 * (d21 ** 2 + x2 ** 2 - x1 ** 2 + y2 ** 2 - y1 ** 2 + z2 ** 2 - z1 ** 2) - d21 * (d23 ** 2 + x2 ** 2 - x3 ** 2 + y2 ** 2 - y3 ** 2 + z2 ** 2 - z3 ** 2)) / (2 * (d21 * y32 - d23 * y12))

    varG = (varE - varB) / (varA - varD)
    varH = (varF - varC) / (varA - varD)
    varI = varA * varG + varB
    varJ = varA * varH + varC

    varK = d02 ** 2 + x0 ** 2 - x2 ** 2 + y0 ** 2 - y2 ** 2 + z0 ** 2 - z2 ** 2 + 2 * x20 * varH + 2 * y20 * varJ
    varL = 2 * (x20 * varG + y20 * varI + z20) # was written wrong in paper 
    varM = 4 * d02 ** 2 * (varG ** 2 + varI ** 2 + 1) - varL ** 2
    varN = 8 * d02 ** 2 * (varG * (x0 - varH) + varI * (y0 - varJ) + z0) + 2 * varL * varK
    varO = 4 * d02 ** 2 * ((x0 - varH) ** 2 + (y0 - varJ) ** 2 + z0 ** 2) - varK ** 2

    pqFt = varN / (2 * varM)
    pqSd = (varN /(2 * varM)) ** 2 - (varO / varM)

    #print("pq-sel", pqFt, pqSd)
    zposCand1 = pqFt + np.sqrt(np.abs(pqSd))
    zposCand2 = pqFt - np.sqrt(np.abs(pqSd))

    #print("z candidates", zposCand1, zposCand2)

    #zpos = zposCand1 if zposCand1 < 0 else zposCand2
    #zpos = np.min([np.max([zposCand1, zposCand2, zBound]), 0])

    ### check which z position might be more accurate depening on last one
    if abs(zposCand1 - lastZPos) < abs(zposCand2 - lastZPos):
        zpos = zposCand1
    else:
        zpos = zposCand2

    ypos = varI * zpos + varJ
    xpos = varG * zpos + varH

    return xpos, ypos, zpos
    #return xpos, ypos, zposCand1, zposCand2

def dist_to_delay(achorPos: list[tuple], vehPos: tuple[float], absolute: bool = False):

    # tau_i = 1/c * (|S - S_0| - |S - S_i|)
    s = np.array(vehPos)
    s0 = np.array(achorPos[0])
    s1 = np.array(achorPos[1])
    s2 = np.array(achorPos[2])
    s3 = np.array(achorPos[3])
    #s4 = np.array(achorPos[4])

    dist0 = np.linalg.norm(s - s0)
    dist1 = np.linalg.norm(s - s1)
    dist2 = np.linalg.norm(s - s2)
    dist3 = np.linalg.norm(s - s3)
    #dist4 = np.linalg.norm(s - s4)

    # S_0 |-----^------------------------|
    # S_1 |-------------^----------------|
    # S_2 |-----------------------^------|
    # tau       0       1         2
    # tau_0i = tau_i - tau_0  

    if absolute:
        t0 = dist0 / SOFSOUND_WATER
        t1 = dist1 / SOFSOUND_WATER
        t2 = dist2 / SOFSOUND_WATER
        t3 = dist3 / SOFSOUND_WATER
        #t4 = dist4 / SOFSOUND_WATER

        return t0, t1, t2, t3

    else:
        tau01 = (dist0 - dist1) / SOFSOUND_WATER
        tau02 = (dist0 - dist2) / SOFSOUND_WATER
        tau03 = (dist0 - dist3) / SOFSOUND_WATER
        #tau04 = (dist0 - dist4) / SOFSOUND_WATER

        return tau01, tau02, tau03

def circlePath(scale: float, offset: float, res: int = 10):
    phi = np.linspace(0, 3*np.pi, res, endpoint=False)
    x = scale * np.sin(phi) + offset
    y = scale * np.cos(phi) + offset

    #z = np.linspace(0, -res, res)
    z = np.array(range(-res, -1))

    # rotate

    return x,y,z

def underwater_localization(anchorsPos: list[tuple], path: tuple[np.ndarray], deg: int = 10, snr: (float | None) = None, simChannel: (str | None) = None):
    """Localization of underwater target via tdoa

    Args
    ----
    anchorsPos: list[tuple]
        absolute 3d Positions of four anchors as (x,y,z)
    path: tuple[np.ndarray]
        list of path coorindates as (x_i ... x_n, y_i ... y_n, z_0 ... z_n)
    deg: int
        degree of pseudorandom code, which influences its total length
    snr: float
        target snr in dB for additive white gaussian noise
    simChannel: str
        channel name which should be used for simulation

    Returns
    -------
    - list of estimated positions
    - list of cfar localization data as (time axis, correlation sum, cfar threshold)
    - tuple of raw recieved signal seperated in its time axis and itself
    - error between real and eastimatet positions
    """

    lstRealPos = []
    lstEstPos = []

    xPos, yPos, zPos = path
    POSITIONS = len(xPos)

    xAnchors = list(zip(*anchorsPos))[0]
    yAnchors = list(zip(*anchorsPos))[1]
    zAnchors = list(zip(*anchorsPos))[2]
    if snr is not None:
        snr = 10 ** (snr / 10)
    fs = 200e3

    totalSumTime = []
    totalSumSignals = []
    lstAnchors = []

    for cx, cy, cz in zip(xPos, yPos, zPos):
        
        absDelays = dist_to_delay(anchorsPos, (cx, cy, cz), absolute=True)
        lstRealPos.append((cx, cy, cz))

        if snr is not None:
            delays, tSigSum, lstAnchors = simulation(absDelays, 4, showAll=False, addGWN=True, useSim=False, polyDeg=deg, targetSNR=snr)
        else:
            delays, tSigSum, lstAnchors = simulation(absDelays, 4, showAll=False, addGWN=False, useSim=False, polyDeg=deg)

        #if simChannel:
        #    delays, tSigSum, lstAnchors = simulation(absDelays, 4, showAll=False, addGWN=False, useSim=True, channel=simChannel, polyDeg=deg, targetSNR=snr)
        #else:
        #    delays, tSigSum, lstAnchors = simulation(absDelays, 4, showAll=False, addGWN=False, useSim=False, polyDeg=deg)

        totalSumSignals = np.append(totalSumSignals, tSigSum[1])

    totalSumTime = np.linspace(0, len(totalSumSignals) / fs, len(totalSumSignals))

    guLen = int(200e3 * 0.00014)
    anchorPeaks = []
    lstSumDelays = []
    lstCFARSig = []

    for i in range(len(lstAnchors)):
        peaks = []

        anchor = np.append(lstAnchors[i], [0] * (len(totalSumSignals) - len(lstAnchors[i])))
        tauCC, tauSigCC = corr_lag(totalSumSignals, anchor, 200e3) 
        cfarSigSum = ca_cfar(tauSigCC, guLen * 4, guLen, 2.1e-1, sort=False, threshold=0.2)

        lstCFARSig.append((tauCC, tauSigCC, cfarSigSum))

        sigCCpks = np.abs(tauSigCC.copy())
        sigCCInd = np.where(sigCCpks > cfarSigSum)

        # group consecutive values for further processing
        groups = group_consecutive_values(sigCCInd[0])

        if len(groups) != POSITIONS - 1:
            exit("Peaks not succefully detected!", len(groups))

        for gr in groups:
            # get maximum of every index group
            cfarPeakInd = np.where(np.abs(tauSigCC) == np.max(np.abs(tauSigCC[gr])))[0][0]
            peaks.append(tauCC[cfarPeakInd])
        
        anchorPeaks.append(peaks)
        
    lastZPos = -1.0
    for pos in range(POSITIONS - 1):
        tdoaSelection = [item[pos] for item in anchorPeaks]
        #lstSumDelays.append(tdoaSelection)
        estPos = quadraticApprox(anchorsPos, tdoaSelection, lastZPos)
        lastZPos = estPos[2]
        lstEstPos.append((estPos[0], estPos[1], estPos[2]))

    posError = totalError(lstRealPos, lstEstPos)

    #for pos in range(POSITIONS - 1):
    #    estPos = quadraticApprox(anchorsPos, lstSumDelays[pos])
    #    lstEstPos.append((estPos[0], estPos[1], estPos[2]))

    #xEstPos = list(zip(*lstEstPos))[0]
    #yEstPos = list(zip(*lstEstPos))[1]
    #zEstPos = list(zip(*lstEstPos))[2]

    return lstEstPos, anchorPeaks, lstCFARSig, (totalSumTime, totalSumSignals), posError

def main():
    POS = 2

    anc = [(15,1,7),(200,10,5),(195,210,6),(16,190,3)]
    testPath = circlePath(30, 100, POS + 1)
    noiseSNR = (-15, -10, -5, 0, 5)
    lstErrors = []

    for snr in noiseSNR:
        eastPositions, lstToas, cfarData, sigCC, error = underwater_localization(anc, testPath, snr=snr)
        lstErrors.append(error)


    ### plotting positions in 3d
    posScatter = go.Figure()

    # plot anchors themselves
    xAnchors = list(zip(*anc))[0]
    yAnchors = list(zip(*anc))[1]
    zAnchors = list(zip(*anc))[2]


    posScatter.add_trace(go.Scatter3d(x=xAnchors, y=yAnchors, z=zAnchors, mode='markers', marker=dict(size=30), name="Anchors", marker_color='#EF553B', text=["S0", "S1", "S2", "S3"], textposition="bottom center"))
    
    # plot real positions of target
    posScatter.add_trace(go.Scatter3d(x=testPath[0], y=testPath[1], z=testPath[2], name="real Position", marker_color='#636EFA'))

    # plot eastimated position of target
    xEstPos = list(zip(*eastPositions))[0]
    yEstPos = list(zip(*eastPositions))[1]
    zEstPos = list(zip(*eastPositions))[2]
    posScatter.add_trace(go.Scatter3d(x=xEstPos, y=yEstPos, z=zEstPos, name="est. Position", marker_color='#FF97FF'))

    posScatter.update_layout(title="Underwater Localization")

    ### plotting correlation toas
    tdoaLineplot = make_subplots(rows=len(anc))

    # plot correlation, cfar
    for ai in range(len(anc)):
        tauCC = cfarData[ai][0]
        tauSigCC = cfarData[ai][1]
        tauCFAR = cfarData[ai][2]

        tdoaLineplot.add_trace(go.Scatter(x=tauCC, y=np.abs(tauSigCC), mode='lines', marker_color='#000'), row=ai+1, col=1)
        tdoaLineplot.add_trace(go.Scatter(x=tauCC, y=tauCFAR, mode='lines', marker_color='#636EFA'), row=ai+1, col=1)

        for t in lstToas[ai]:
            tdoaLineplot.add_vline(t, line_color='#EF553B', line_width=3, line_dash='solid', row=ai+1, col=1)

    for pos in range(POS - 1):

        selection = [item[pos] for item in lstToas]
        for i in range(len(anc)):
            tdoaLineplot.add_vline(np.min(selection), line_color='#7F7F7F', line_width=3, line_dash='dash', row=i+1, col=1)

    tdoaLineplot.update_layout(showlegend=False, title="CFAR TOA correlation peaks")
    

    errorPlot = make_subplots(rows=len(noiseSNR))
    for i in range(len(lstErrors)):
        title_name = "Error for " + str(noiseSNR[i]) + " dB"
        errorPlot.add_trace(go.Scatter(x=list(range(POS)), y=lstErrors[i], name=title_name), row=i+1, col=1)
    errorPlot.update_layout(title="Error between real and east. positions")

    ### show all plots
    posScatter.show()
    tdoaLineplot.show()
    errorPlot.show()

if __name__ == "__main__":
    main()


"""
def alt():
    global file_index, load_index
    #lstSNRdB = [-15, -10, -5, 0, 5]
    lstSNRdB = [5]
    #chns = ['PLANE1', 'PLANE2', 'CASTLE1', 'CASTLE2', 'CASTLE3']
    degs = [8, 9, 10]
    

    #anc = [(np.random.randint(0,100), np.random.randint(0,100),np.random.randint(-50,0)), 
    #    (np.random.randint(0,100), np.random.randint(0,100),np.random.randint(-50,0)), 
    #    (np.random.randint(0,100), np.random.randint(0,100),np.random.randint(-50,0)), 
    #    (np.random.randint(0,100), np.random.randint(0,100), np.random.randint(-50,0))]

    #anc = [(10,1,np.random.randint(5,20)), (100,10,np.random.randint(5,20)), (10,100,np.random.randint(5,20)), (100,100,np.random.randint(5,20)), (50,50,np.random.randint(5,20))]
    #anc = [(10,1,1),(100,10,1),(100,100,1),(10,100,1), (10,50,1)]
    anc = [(15,1,7),(200,10,5),(195,210,6),(16,190,3)]
    #anc = [(15,1,-0.5), (70,10,-0.5), (50,60,-0.5), (16,95,-0.5)]
    #anc = [(0,0,-0.5),(30,0,-0.5),(20,20,-0.5),(0,30,-0.5)]
    #print("Real delays", realDelays)

    lstRealPos = []
    lstEstPos = []
    lstSumEstPos = []

    xPos, yPos, zPos = circlePath(30, 100, POSITIONS)
    

    for ch in chns:
        snr = 10 ** (snr / 10)
        maxDelays, cfarDelays = simulation(realDelays, 3, showAll=False, addGWN=False, targetSNR=snr, useSim=True, channel=ch, polyDeg=10)
        print("Detected delays by max:", maxDelays)
        print("Detected delays by cfar:", cfarDelays)
        file_index = 0
        load_index = 0

        estPos = localLateration(anc, realDelays)
        print("new pos", estPos)

    #showButterBode()
    #showRaisedCosine()

    posscatter = go.Figure()

    xAnchors = list(zip(*anc))[0]
    yAnchors = list(zip(*anc))[1]
    zAnchors = list(zip(*anc))[2]

    posscatter.add_trace(go.Scatter3d(x=xAnchors, y=yAnchors, z=zAnchors, mode='markers', marker=dict(size=30), name="Anchors", marker_color='#EF553B', text=["S0", "S1", "S2", "S3", "S4"], textposition="bottom center"))
    snr = 10
    # sweet spot -11dB, everyting above creates intense distortion
    snr = 10 ** (snr / 10)

    waypoint_index = 0

    totalSumTime = []
    totalSumSignals = []
    lastTime = 0
    lstAnchors = []

    fs = 200e3
    signalWaitTime = 0.2

    for cx, cy, cz in zip(xPos, yPos, zPos):

        print("pos", waypoint_index)

        relDelays = dist_to_delay(anc, (cx, cy, cz))
        absDelays = dist_to_delay(anc, (cx, cy, cz), absolute=True)

        lstRealPos.append((cx, cy, cz))
        
        delays, tSigSum, lstAnchors = simulation(absDelays, 4, showAll=False, addGWN=True, useSim=True, channel='CASTLE1', polyDeg=10, targetSNR=snr)



        #totalSumTime = np.append(totalSumTime, tSigSum[0])
        #print("length of time", len(tSigSum[0]))
        #totalSumTime[:] += lastTime
        #print(lastTime)
        #lastTime = tSigSum[0][-1]


        ### shorting signal for selected time window
        #totalSumSignals = np.append(totalSumSignals, tSigSum[1][int(fs*signalWaitTime):])

        ### shortining signal from simualtor
        totalSumSignals = np.append(totalSumSignals, tSigSum[1][:int(fs*signalWaitTime)])
        #totalSumSignals = np.append(totalSumSignals, tSigSum[1])

        #print("rel delays from direct calc", round(relDelays[0],3), round(relDelays[1],3), round(relDelays[2],3))
        #print("rel. delays from simulation", round(maxDelays[0],3), round(maxDelays[1],3), round(maxDelays[2],3))

        #estPos = localLateration(anc, maxDelays)

        estPos = quadraticApprox(anc, delays[0])
    

        lstEstPos.append((estPos[0], estPos[1], estPos[2]))
        #lstEstPos.append((estPos[0], estPos[1], estPos[3])) #shows that alt solution is also in negative space

        waypoint_index += 1

    totalSumTime = np.linspace(0, len(totalSumSignals) / fs, len(totalSumSignals))



    posscatter.add_trace(go.Scatter3d(x=xPos, y=yPos, z=zPos, name="real Position", marker_color='#636EFA'))

    xEstPos = list(zip(*lstEstPos))[0]
    yEstPos = list(zip(*lstEstPos))[1]
    zEstPos = list(zip(*lstEstPos))[2]

    posscatter.add_trace(go.Scatter3d(x=xEstPos, y=yEstPos, z=zEstPos, name="est. MAX Position", marker_color='#AB63FA'))

    #posscatter.show()

    fig = go.Figure()

    localError = totalError(lstRealPos, lstEstPos)

    fig.add_trace(go.Scatter(y=localError))
    fig.show()


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=totalSumTime, y=np.real(totalSumSignals)))
    fig.show()


    figure = make_subplots(rows=4, cols=1)
    guLen = int(200e3 * 0.00014)

    anchorPeaks = []
    lstSumDelays = []


    for i in range(len(lstAnchors)):

        peaks = []
        anchor = np.append(lstAnchors[i], [0] * (len(totalSumSignals) - len(lstAnchors[i])))
        tauCC, tauSigCC = corr_lag(totalSumSignals, anchor, 200e3)


        varSigSum = ca_cfar(tauSigCC, guLen * 6, guLen, 0.2e-1, sort=False)
        
        SigCCpks = np.abs(tauSigCC.copy())
        sigCCind = np.where(SigCCpks > varSigSum)

        figure.add_trace(go.Scatter(x=tauCC, y=np.abs(tauSigCC), mode='lines', marker_color='#000'), row=i+1, col=1)
        figure.add_trace(go.Scatter(x=tauCC, y=varSigSum, mode='lines', marker_color='#636EFA'), row=i+1, col=1)

        lastVal = 0

        groups = group_consecutive_values(sigCCind[0])

        #for j in sigCCind[0]:
        #    
        #    if j != lastVal + 1:
        #        figure.add_vline(tauCC[j], line_color='#EF553B', line_width=3, line_dash='solid', row=i+1, col=1)
        #        peaks.append(tauCC[j])
        #    lastVal = j
        for gr in groups:

            cfarPeakInd = np.where(np.abs(tauSigCC) == np.max(np.abs(tauSigCC[gr])))[0][0]
            print("found peak at index ", cfarPeakInd)

            figure.add_vline(tauCC[cfarPeakInd], line_color='#EF553B', line_width=3, line_dash='solid', row=i+1, col=1)
            peaks.append(tauCC[cfarPeakInd])

        #peaks = [pk + 5e-5 for pk in peaks]

        anchorPeaks.append(peaks)


    for pos in range(POSITIONS - 1):

        selection = [item[pos] for item in anchorPeaks]
        lstSumDelays.append(selection)
        print("pos. " + str(pos), selection)
        for i in range(len(lstAnchors)):
            figure.add_vline(np.min(selection), line_color='#7F7F7F', line_width=3, line_dash='dash', row=i+1, col=1)


    print("peaks", anchorPeaks)
        

    figure.update_layout(showlegend=False)
    figure.show()

    ### seconds position loop for summed correlation eastimates


    for pos in range(POSITIONS - 1):

        estPos = quadraticApprox(anc, lstSumDelays[pos])
        lstSumEstPos.append((estPos[0], estPos[1], estPos[2]))

    xEstPos = list(zip(*lstSumEstPos))[0]
    yEstPos = list(zip(*lstSumEstPos))[1]
    zEstPos = list(zip(*lstSumEstPos))[2]

    posscatter.add_trace(go.Scatter3d(x=xEstPos, y=yEstPos, z=zEstPos, name="est. CFAR Position", marker_color='#FF97FF'))
    posscatter.show()
def leastSquareEst(anchorPos: list[tuple], delays: list[float]):

    m12 = delays[0] * SOFSOUND_WATER
    m13 = delays[0] * SOFSOUND_WATER
    m14 = delays[0] * SOFSOUND_WATER

    dX2 = anchorPos[1][0] - anchorPos[0][0]
    dX3 = anchorPos[2][0] - anchorPos[0][0]
    dX4 = anchorPos[3][0] - anchorPos[0][0]

    dY2 = anchorPos[1][1] - anchorPos[0][1]
    dY3 = anchorPos[2][1] - anchorPos[0][1]
    dY4 = anchorPos[3][1] - anchorPos[0][1]

    dZ2 = anchorPos[1][2] - anchorPos[0][2]
    dZ3 = anchorPos[2][2] - anchorPos[0][2]
    dZ4 = anchorPos[3][2] - anchorPos[0][2]

    dK2 = dX2 ** 2 + dY2 ** 2 + dZ2 ** 2
    dK3 = dX3 ** 2 + dX3 ** 2 + dZ3 ** 2
    dK4 = dX4 ** 2 + dY4 ** 2 + dZ4 ** 2

    matA = np.array([[dX2, dY2, dZ2], [dX3, dY3, dZ3], [dX3, dY4, dZ4]])

    vecC = np.array([dK2 - m12 ** 2, dK3 - m13 ** 2, dK4 - m14 ** 2])
    vecD = np.array([-m12, -m13, -m14])
    vecB = vecC + vecD

    dPos = np.linalg.lstsq(matA, vecB, rcond=None)[0]
    
    return dPos"""

"""def localLateration(anchorPos: list[tuple], delays: list[float]):

    x0, y0, z0 = anchorPos[0]
    x1, y1, z1 = anchorPos[1]
    x2, y2, z2 = anchorPos[2]
    x3, y3, z3 = anchorPos[3]
    x4, y4, z4 = anchorPos[4]

    tau01 = delays[0]
    tau02 = delays[1]
    tau03 = delays[2]
    tau04 = delays[3]

    d01 = SOFSOUND_WATER * tau01
    d02 = SOFSOUND_WATER * tau02
    d03 = SOFSOUND_WATER * tau03
    d04 = SOFSOUND_WATER * tau04

    #matA = np.array(
    #    [[x0 - x1, y0 - y1, z0 - z1, d01], 
    #     [x0 - x2, y0 - y2, z0 - z2, d02], 
    #     [x0 - x3, y0 - y3, z0 - z3, d03], 
    #     [x0 - x4, y0 - y4, z0 - z4, d04]]
    
    vecB = np.array(
        [x0 ** 2 - x1 ** 2 + y0 ** 2 - y1 ** 2 + z0 ** 2 - z1 ** 2 + d01 ** 2, 
         x0 ** 2 - x2 ** 2 + y0 ** 2 - y2 ** 2 + z0 ** 2 - z2 ** 2 + d02 ** 2, 
         x0 ** 2 - x3 ** 2 + y0 ** 2 - y3 ** 2 + z0 ** 2 - z3 ** 2 + d03 ** 2,
         x0 ** 2 - x4 ** 2 + y0 ** 2 - y4 ** 2 + z0 ** 2 - z4 ** 2 + d04 ** 2]
    ) / 2

    matA = np.array(
        [[x0 - x1, y0 - y1, z0 - z1, d01], 
         [x0 - x2, y0 - y2, z0 - z2, d02], 
         [x0 - x3, y0 - y3, z0 - z3, d03]]
    )
    vecB = np.array(
        [x0 ** 2 - x1 ** 2 + y0 ** 2 - y1 ** 2 + z0 ** 2 - z1 ** 2 + d01 ** 2, 
         x0 ** 2 - x2 ** 2 + y0 ** 2 - y2 ** 2 + z0 ** 2 - z2 ** 2 + d02 ** 2, 
         x0 ** 2 - x3 ** 2 + y0 ** 2 - y3 ** 2 + z0 ** 2 - z3 ** 2 + d03 ** 2]
    ) / 2

    ### positions by last sqare regression using direct inverse method (A^T * A)^-1 * A^T * b
    # source https://math.stackexchange.com/questions/1722021/trilateration-using-tdoa
    #pos = np.dot(np.dot(np.linalg.inv(np.dot(matA.T, matA)), matA.T), vecB)[:3]
    #pos = np.linalg.solve(matA, vecB)
    #pos, resid, _, sing = np.linalg.lstsq(matA, vecB, rcond=-1)
    #pos, resid, _, sing = linalg.lstsq(matA, vecB)

    result = optimize.lsq_linear(matA, vecB)

    #print("residuals", resid, "singular vals", sing)

    # least suqare/quadratic solution for using only 4 anchors
    # *-----|-----* 

    return result.x"""