from processing import simulation
import plotly.graph_objects as go
import numpy as np

# source http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/Soundv.html
SOFSOUND_WATER = 1493

def localLateration(anchorPos: list[tuple], delays: list[float]):

    x0, y0 = anchorPos[0]
    x1, y1 = anchorPos[1]
    x2, y2 = anchorPos[2]

    tau01 = delays[1] - delays[0]
    tau02 = delays[2] - delays[0]

    d01 = SOFSOUND_WATER * tau01
    d02 = SOFSOUND_WATER * tau02

    matA = np.array([[x0 - x1, y0 - y1, d01], [x0 - x2, y0 - y2, d02]])
    vecB = np.array([x0 ** 2 - x1 ** 2 + y0 ** 2 - y1 ** 2 + d01 ** 2, x0 ** 2 - x2 ** 2 + y0 ** 2 - y2 ** 2 + d02 ** 2]) / 2

    ### positions by last sqare regression using direct inverse method (A^T * A)^-1 * A^T * b
    # source https://math.stackexchange.com/questions/1722021/trilateration-using-tdoa
    #pos = np.dot(np.dot(np.linalg.inv(np.dot(matA.T, matA)), matA.T), vecB)[:2]
    pos = np.linalg.lstsq(matA, vecB, rcond=None)[0][:2]

    return pos

def distToDelay(achorPos: list[tuple], startPos: tuple[float]):

    # tau_i = 1/c * (|S - S_0| - |S - S_i|)

    d0 = np.linalg.norm(np.asarray(startPos) - np.asarray(achorPos[0]))
    d01 = np.linalg.norm(np.asarray(achorPos[0]) - np.asarray(achorPos[1]))
    d02 = np.linalg.norm(np.asarray(achorPos[0]) - np.asarray(achorPos[2]))

    # S_0 |-----^------------------------|
    # S_1 |-------------^----------------|
    # S_2 |-----------------------^------|
    # tau       0       1         2
    # tau_0i = tau_i - tau_0  
    tau01 = (d0 - d01) / SOFSOUND_WATER
    tau02 = (d0 - d02) / SOFSOUND_WATER

    tau0 = d0 / SOFSOUND_WATER
    tau1 = tau0 + tau01
    tau2 = tau0 + tau02

    return tau0, tau1, tau2


def circlePath(scale: float, offset: float, res: int = 10):
    phi = np.linspace(0, 2*np.pi, res)
    x = scale * np.sin(phi) + offset
    y = scale * np.cos(phi) + offset

    return x,y

def main():
    global file_index, load_index
    #lstSNRdB = [-15, -10, -5, 0, 5]
    lstSNRdB = [5]
    #chns = ['PLANE1', 'PLANE2', 'CASTLE1', 'CASTLE2', 'CASTLE3']
    degs = [8, 9, 10]
    snr = 0

    anc = [(0,0), (0,50), (50,25)]
    
    #print("Real delays", realDelays)

    lstRealDelays = []
    lstEstPos = []

    xPos, yPos = circlePath(10, 20, 30)

    """for ch in chns:
        snr = 10 ** (snr / 10)
        maxDelays, cfarDelays = simulation(realDelays, 3, showAll=False, addGWN=False, targetSNR=snr, useSim=True, channel=ch, polyDeg=10)
        print("Detected delays by max:", maxDelays)
        print("Detected delays by cfar:", cfarDelays)
        file_index = 0
        load_index = 0

        estPos = localLateration(anc, realDelays)
        print("new pos", estPos)"""

    #showButterBode()
    #showRaisedCosine()

    fig = go.Figure()

    xAnchors = list(zip(*anc))[0]
    yAnchors = list(zip(*anc))[1]


    fig.add_trace(go.Scatter(x=xAnchors, y=yAnchors, mode='markers', marker=dict(size=30), name="Anchors", marker_color='#EF553B', text=["S0", "S1", "S2"], textposition="bottom center"))
    #fig.add_trace(go.Scatter(x=[realPos[0]], y=[realPos[1]], mode='markers', marker=dict(size=50), name="real Position", marker_color='#636EFA'))
    #fig.add_trace(go.Scatter(x=[estPos[0]], y=[estPos[1]], mode='markers', marker=dict(size=50), name="est. Position", marker_color='#AB63FA'))

    for cx, cy in zip(xPos, yPos):

        realDelays = distToDelay(anc, (cx, cy))
        lstRealDelays.append((cx, cy))

        maxDelays, cfarDelays = simulation(realDelays, 3, showAll=False, addGWN=False, targetSNR=snr, useSim=True, channel='PLANE1', polyDeg=10)

        #print("real delays", realDelays)
        #print("max delays", maxDelays)

        estPos = localLateration(anc, maxDelays)
        print("est. Pos", estPos)
        lstEstPos.append((estPos[0], estPos[1]))

    fig.add_trace(go.Scatter(x=xPos, y=yPos, name="real Position", marker_color='#636EFA'))

    xEstPos = list(zip(*lstEstPos))[0]
    yEstPos = list(zip(*lstEstPos))[1]

    fig.add_trace(go.Scatter(x=xEstPos, y=yEstPos, name="est. Position", marker_color='#AB63FA'))

    fig.show()



if __name__ == "__main__":
    main()
    