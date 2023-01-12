import processing as pr

def main():
    pr.simulation([5e-3, 10e-3], 2, showAll=True, addGWN=False, useSim=False, polyDeg=10, labExport=True)
    #pr.simulation([5e-3, 10e-3], 2, showAll=False, addGWN=False, useSim=False, polyDeg=9, labExport=True)
    #pr.simulation([5e-3, 10e-3], 2, showAll=False, addGWN=False, useSim=False, polyDeg=8, labExport=True)
    # use filt (not filtfilt) to fitler the signal by fitler b and hydrophone vals
    # gated periods
    # 1.5V
if __name__ == "__main__":
    main()
    