from processing import simulation
from scipy.optimize import minimize
import plotly.graph_objects as go
import numpy as np

# source http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/Soundv.html
SOFSOUND_WATER = 1493

def localLateration(anchorPos: list[tuple], delays: list[float]):

    x0, y0 = anchorPos[0]
    x1, y1 = anchorPos[1]
    x2, y2 = anchorPos[2]

    tau01 = delays[0] 
    tau02 = delays[1] 

    d01 = SOFSOUND_WATER * tau01
    d02 = SOFSOUND_WATER * tau02

    matA = np.array([[x0 - x1, y0 - y1, d01], [x0 - x2, y0 - y2, d02]])
    vecB = np.array([x0 ** 2 - x1 ** 2 + y0 ** 2 - y1 ** 2 + d01 ** 2, x0 ** 2 - x2 ** 2 + y0 ** 2 - y2 ** 2 + d02 ** 2]) / 2

    ### positions by last sqare regression using direct inverse method (A^T * A)^-1 * A^T * b
    # source https://math.stackexchange.com/questions/1722021/trilateration-using-tdoa
    pos = np.dot(np.dot(np.linalg.inv(np.dot(matA.T, matA)), matA.T), vecB)[:2]

    return pos

def distToDelay(achorPos: list[tuple], startPos: tuple[float]):

    d0 = np.linalg.norm(np.asarray(achorPos[0]) - np.asarray(startPos))
    d1 = np.linalg.norm(np.asarray(achorPos[1]) - np.asarray(startPos))
    d2 = np.linalg.norm(np.asarray(achorPos[2]) - np.asarray(startPos))

    tau0 = d0 / SOFSOUND_WATER
    tau1 = d1 / SOFSOUND_WATER
    tau2 = d2 / SOFSOUND_WATER

    return tau0, tau1, tau2


def main():
    global file_index, load_index
    #lstSNRdB = [-15, -10, -5, 0, 5]
    lstSNRdB = [5]
    #chns = ['PLANE1', 'PLANE2', 'CASTLE1', 'CASTLE2', 'CASTLE3']
    chns = ['PLANE1']
    degs = [8, 9, 10]
    snr = 0

    realPos = (30,33)
    anchors = [(0,0), (0,50), (50,25)]
    realDelays = distToDelay(anchors, realPos)
    print("Real delays", realDelays)

    for ch in chns:
        snr = 10 ** (snr / 10)
        maxDelays, cfarDelays = simulation(realDelays, 3, showAll=False, addGWN=False, targetSNR=snr, useSim=True, channel=ch, polyDeg=10)
        print("Detected delays by max:", maxDelays)
        print("Detected delays by cfar:", cfarDelays)
        file_index = 0
        load_index = 0

        anc = [(0,0), (0,50), (50,25)]

        relativeDelays = [maxDelays[1] - maxDelays[0], maxDelays[2] - maxDelays[0]]

        estPos = localLateration(anc, relativeDelays)
        print("new pos", estPos)

    #showButterBode()
    #showRaisedCosine()

    fig = go.Figure()

    xAnchors = list(zip(*anc))[0]
    yAnchors = list(zip(*anc))[1]

    fig.add_trace(go.Scatter(x=xAnchors, y=yAnchors, mode='markers', marker=dict(size=50), name="Anchors", marker_color='#EF553B'))
    fig.add_trace(go.Scatter(x=[realPos[0]], y=[realPos[1]], mode='markers', marker=dict(size=50), name="real Position", marker_color='#636EFA'))
    fig.add_trace(go.Scatter(x=[estPos[0]], y=[estPos[1]], mode='markers', marker=dict(size=50), name="est. Position", marker_color='#AB63FA'))

    fig.show()

if __name__ == "__main__":
    main()
    