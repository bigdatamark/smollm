# [MODULE DOC] KL divergence between two empirical distributions of values in [0, 1], estimated with
# histograms.
# `NB_BINS` (40) is the default number of bins.
import numpy as np


NB_BINS = 40


def kl_div(p, q, nb_bins=NB_BINS):
    # [FUNCTION DOC] Return KL(p || q) estimated from density histograms (`nb_bins` bins over [0, 1]) of the
    # value arrays p and q:
    # sum over bins of fp * log(fp / fq), with NaN terms and bins where fq == 0 counted as 0. Values outside
    # [0, 1] are ignored. May emit numpy divide warnings.
    # KNOWN ISSUE: bins with fp > 0 and fq == 0 should make the divergence infinite but are silently dropped.
    freq_p, _ = np.histogram(p, bins=nb_bins, range=(0.0, 1.0), density=True)
    freq_q, _ = np.histogram(q, bins=nb_bins, range=(0.0, 1.0), density=True)
    elem = freq_p * np.log(freq_p / freq_q)
    return np.sum(np.where((~np.isnan(elem)) & (freq_q != 0), elem, 0))
