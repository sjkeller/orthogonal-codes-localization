keller_orthogonal-codes
===============

# Pseudo Random Binary Sequence (PRBS)

## properties

- **balanced**
    false and true are equally distributed
- **sub-sequence length limit**
    subsequences consisting of only false and true are limited in theire length
- **orthogonal cross-correlation**
    correlation of even small delayed sequences is extremly small, where otherwise the correlation of exactly lined up ones is huge
 
## Implementation

### Linear Feedback Shift Register (LFSR)

used for generating pseudo random numbers

x1 -> x2 -> x3 -> x4 -> OUT
↑           ↓     ↓
└---------- mod 2 (+)

## Orthogonality of shifted PRBS

- Because of the nature of random signals the resulting spectrum should be a constant signal
- the inverse fourier transform of a constant frequency spectrum would be a dirac which is impossible to implement
- so this is why pseudo random numers are used to at least converge to a constant spectrum
- we know such signals have interesting cross-correlation characters (see section properties)
- this implies that even extremly small delayed PN codes are **orthogonal** to each other
- thus we should be able to detect small delays with the help or correlation
