import numpy as np


def consWeight_increase(epo, args):
    if args.consWeight_start: epo = max(0, epo - args.consWeight_start)
    return 0. if epo == 0 else _value_increase(epo, args.consWeight_max, args.consWeight_min, args.consWeight_rampup)


def pseudoWeight_increase(epo, args):
    if args.pseudoWeight_start: epo = max(0, epo - args.pseudoWeight_start)
    return 0. if epo == 0 else _value_increase(epo, args.pseudoWeight_max, args.pseudoWeight_min, args.pseudoWeight_rampup)


def FDLWeight_increase(epo, args):
    return _value_increase(epo, args.FDLWeight_max, args.FDLWeight_min, args.FDLWeight_rampup)


def FDLWeight_decrease(epo, args):
    return _value_decrease(epo, args.FDLWeight_max, args.FDLWeight_min, args.FDLWeight_rampup)


def _value_increase(epo, maxValue, minValue, rampup):
    return minValue + (maxValue - minValue) * _sigmoid_rampup(epo, rampup)


def _value_decrease(epo, maxValue, minValue, rampup):
    return minValue + (maxValue - minValue) * (1.0 - _sigmoid_rampup(epo, rampup))


def _sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))