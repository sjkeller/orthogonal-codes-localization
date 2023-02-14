# Orthogonal Codes for Acoustic Underwater Localization - Scripts

Python scripts accompanying the **Orthogonal Codes for Acoustic Underwater Localization** research project. The Scripts should be compatible with python 3.10 and upwards. Following dependecies are needed for running:

### Used packages

- **numpy**
: numeric computing library containing a lot of neat array structures and light processing algorithms
    [Documentation](https://numpy.org/doc/stable/) [PyPI](https://pypi.org/project/numpy/)

- **scipy**
: scientific comupting library with a lot of matlab like functions
    [Documentation](https://docs.scipy.org/doc/scipy/) [PyPI](https://pypi.org/project/scipy/)

- **commpy**
: signal processing library containing multiple algorithms for digital communication
    [Documentation](https://commpy.readthedocs.io/en/latest/) [PyPI](https://pypi.org/project/scikit-commpy/)

- **plotly**
: sophisticated plotting library for python and a lot of other programming languages
    [Documentation](https://plotly.com/python/) [PyPI](https://pypi.org/project/plotly/)

### Installation
`pip install numpy scipy scikit-commpy plotly`

## Code implementation

- `polys.py`
: source of polynomials for m-sequences

- `sequences`
: m-sequences, kasami and gold code implementation

## Code comparison and evaluation

- `comparison.py`
: comparse m-sequence, gold and kasami codes via PSR and ACR

- `processing.py`
: signal processing including some parts of localization

- `localization.py`
: localization of simulated 3D points with 4 anchors

- `labtest.py`
: real world application of localuzation with 2 anchors