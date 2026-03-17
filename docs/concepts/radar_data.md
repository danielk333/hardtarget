# Radar data

The analysis always assumes the radar data consists of a
continuous stream of signal samples. If in reality there was no samples taken
during an interval the sample stream should be zero-padded during this time.
As such the signal sample stream is split up into regular cycles. Each cycle
occurs back-to-back and the time a cycle takes is called an inter-pulse-period,
or IPP.

Every analysis consists of one or several received signals (RX) and can also contain a
transmitted signal (TX). If no transmitted signal is available it should be simulated.
These signals can either be superimposed in the same channel or be in different
channels in the data structure. Either way, there are several segments within
each cycle that we need to extract and index. Hence, there are three main levels
of signal indices within one cycle, which we will call Index Levels (IL's):

```text
    0) Signal samples (can be same channel)
        - RX signal (size=IPP length)
        - TX signal (size=IPP length)
    0d) Decimated signal samples
        - RX signal (size=IPP length / decimation)
    1) Stenciled samples:
        - RX window (size=chosen reception length, i.e. all range gates)
        - TX pulse (size=length of transmitted pulse)
    2) Target range-gate:
        - RX pulse (size=length of transmitted pulse, offset by chosen range gate)
```

The IL-0d is a bit of a special case as temporal decimation (piece-wise sums of
neighbors) only maintains desirable properties (both statistical and signal
vise) if it is done on an isochronal and continuous sample stream. As such,
decimation operation only occurs on a level 0 signal.

Illustration of the above:

```text

    IL-0     : |0123456789...................................|
    Signal   : |xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|
    IL-0d    : | 0  1  2  3  4  5  6  7  8  9  .  .  .  .  . |
    Decimated: | x  x  x  x  x  x  x  x  x  x  x  x  x  x  x |
    IL-1 tx  : |  012345                                     |
    TX pulse : |--xxxxxx-------------------------------------|
    IL-1 rx  : |                0123456789..............     |
    RX window: |----------------xxxxxxxxxxxxxxxxxxxxxxxx-----|
    IL-2     : |                           012345            |
    RX pulse : |---------------------------xxxxxx------------|
```

In the above example, the first sample of the TX pulse would have index 2 in
terms of IL-0 but would have index 0 in terms of IL-1. The first sample of IL-0d
would represent samples 0, 1, and 2 in IL-0. Similarly the first sample of the
RX pulse would have index 27 in terms of IL-0, index 11 in IL-1 and index 0 in
IL-2. Also, index 0, 1, and 2 in the RX pulse IL-2 would be associated with only
index 9 in IL-0d.

The general strategy for decimation is to zero-pad the target vector if the
decimation does not evenly divide the vector.

In the analysis, several of these cycles are stacked on top of each other. This
means that when selecting the IL-1 tx samples the resulting array is no longer
isochronal and continues but instead has a jump in the middle, e.g.

```text
    IL-1 tx  : |  012345        6789..        ......      |
    TX pulse : |--xxxxxx--------xxxxxx--------xxxxxx------|
```

Since range from a transmitter converted to time is measured from the start of
transmission (i.e. the time between when the leading edge of the wave leaves the
transmitter and reaches the receiver), the range gates are also always measured
in samples relative the start of the TX pulse + 1, in terms of IL-0. Which means
that sample of range-gate 0 has traveled the time of 1 sample. The largest range
gate is at the end of reception, which basically means only one sample of the TX
could be measured.
