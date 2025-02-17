"""
Digital communication signal simulation tools
"""
import numpy as np
from numpy.random import Generator, MT19937, SeedSequence
from more_itertools import windowed


class BitstreamGenerator(object):
    """
    Generates a random bit-stream
    """

    def __init__(self,
                 **kwargs):
        """
        BitstreamGenerator class constructor

        Parameters
        ----------
        self: BitstreamGenerator
            BitstreamGenerator class object handle

        kwargs: dict
            Optional parameters

        Returns
        ----------
        self: BitstreamGenerator
            BitstreamGenerator class object handle
        """
        entropy =\
            kwargs.get("entropy",
                       3564527794581704992655380779498392576)

        sseq = SeedSequence(entropy=entropy)
        self.rng = Generator(MT19937(sseq))

    def __call__(self,
                 nsamples):
        """
        Generates an [nsamples] random bit stream

        Parameters
        ----------
        self: BitstreamGenerator
            BitstreamGenerator class object handle

        nsamples: int
            Number of bits to generate

        Returns
        ----------
        bit_stream: numpy.ndarray
            Random bit stream
        """
        return (self.rng.uniform(size=nsamples) > 0.5).astype(int)


class AWGNGenerator(object):
    """
    Additive White Gaussian Noise (AWGN) simulator
    """

    def __init__(self,
                 snr_db,
                 **kwargs):
        """
        AWGNGenerator class contstructor

        Parameters
        ----------
        self: AWGNGenerator class object reference

        snr_db: float
            Signal to Noise Ratio [dB]

        kwargs: dict
            Optional parameters

        Returns
        -------
        self: AWGNGenerator class object reference
        """
        entropy =\
            kwargs.get("entropy",
                       16429983695829393284219889283603727513)

        sseq = SeedSequence(entropy=entropy)
        self.rng = Generator(MT19937(sseq))

        signal_var = kwargs.get("signal_var", 1.0)
        self.noise_var = signal_var * 10 ** (-snr_db / 10)

    def __call__(self,
                 nsamples):
        """
        Generates AWGN noise samples

        Parameters
        ----------
        self: AWGNGenerator class object reference

        nsamples: int
            Number of AWGN samples to generate

        Returns
        -------
        awgn: numpy.ndarray
            AWGN samples
        """
        awgn =\
            self.rng.normal(size=nsamples) +\
            1j * self.rng.normal(size=nsamples)

        return awgn * np.sqrt(self.noise_var) / np.std(awgn)


class BPSKGenerator(object):
    """
    Binary Phase Shift Keying (BPSK) digital communication signal
    generator
    """

    def __init__(self,
                 snr_db,
                 **kwargs):
        """
        BPSKGenerator class contstructor

        Parameters
        ----------
        self: BPSKGenerator class object reference

        snr_db: float
            Signal to Noise Ratio [dB]

        kwargs: dict
            Optional parameters

        Returns
        -------
        self: BPSKGenerator class object reference
        """
        self.bitstream_gen = BitstreamGenerator(**kwargs)

        self.awgn_gen = AWGNGenerator(snr_db,
                                      **kwargs)

        self.samples_per_symbol =\
            kwargs.get("samples_per_symbol", 8)

    def __call__(self,
                 nsymbols):
        """
        Simulates the requested number of BPSK symbols

        Parameters
        ----------
        self: BPSKGenerator class object reference

        nsymbols: int
            Number of BPSK symbols to generate

        Returns
        -------
        bpsk_signal: numpy.ndarray
            BPSK signal samples
        """
        random_bits = self.bitstream_gen(nsymbols)

        bpsk_state =\
            [(-1 if elem > 0.5 else 0) * np.ones(self.samples_per_symbol)
             for elem in random_bits]

        bpsk_signal = np.exp(np.concatenate(bpsk_state) * 1j * np.pi)
        bpsk_signal = bpsk_signal / np.std(bpsk_signal)

        return bpsk_signal + self.awgn_gen(len(bpsk_signal))


class QPSKGenerator(object):

    def __init__(self,
                 snr_db,
                 **kwargs):
        """
        QPSKGenerator class contstructor

        Parameters
        ----------
        self: QPSKGenerator class object reference

        snr_db: float
            Signal to Noise Ratio [dB]

        kwargs: dict
            Optional parameters

        Returns
        -------
        self: QPSKGenerator class object reference
        """
        self.bitstream_gen = BitstreamGenerator(**kwargs)

        self.awgn_gen = AWGNGenerator(snr_db,
                                      **kwargs)

        self.samples_per_symbol =\
            kwargs.get("samples_per_symbol", 8)

    def __call__(self,
                 nsymbols):
        """
        Simulates the requested number of QPSK symbols

        Parameters
        ----------
        self: QPSKGenerator class object reference

        nsymbols: int
            Number of BPSK symbols to generate

        Returns
        -------
        qpsk_signal: numpy.ndarray
            QPSK signal samples
        """
        random_bits = self.bitstream_gen(2 * nsymbols)

        qpsk_symbols =\
            np.array([2*elem[0] + elem[1]
                      for elem in windowed(random_bits,
                                           n=2,
                                           step=2)])

        qpsk_phase_states =\
            np.array([elem * np.pi / 2 for elem in qpsk_symbols])

        qpsk_signal =\
            [1j * elem * np.ones(self.samples_per_symbol)
             for elem in qpsk_phase_states]

        qpsk_signal = np.exp(np.concatenate(qpsk_signal))

        return qpsk_signal + self.awgn_gen(len(qpsk_signal))
