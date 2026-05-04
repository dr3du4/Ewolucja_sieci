"""
Sensing Proxy for UAV-ISAC Simulation.

Simplified sensing model for target detection and localization.
Computes beampattern gain, sensing SNR, and Cramér-Rao Bound (CRB)
for range estimation.

References:
    Liu et al., "Fair ISAC for Multi-UAV-Enabled IoT," IEEE IoT Journal, 2024.
    Jing et al., "ISAC From the Sky," IEEE TWC, 2024.
"""

import numpy as np
from dataclasses import dataclass
from .channel_model import ChannelModel, ChannelParams


@dataclass
class SensingParams:
    """Sensing subsystem parameters."""
    radar_cross_section: float = 1.0    # RCS of target [m²]
    n_antennas: int = 8                 # Number of antenna elements
    pulse_width: float = 1e-6           # Pulse width [s]
    n_pulses: int = 64                  # Pulses per CPI
    sensing_power_ratio: float = 0.5    # Fraction of power allocated to sensing
    noise_power_dbm: float = -104.0     # Noise power [dBm]


class SensingModel:
    """
    Simplified radar sensing model for UAV-ISAC.

    Models:
    - Beampattern gain toward targets
    - Sensing SNR (two-way path loss)
    - CRB for range estimation
    """

    SPEED_OF_LIGHT = 3e8

    def __init__(
        self,
        sensing_params: SensingParams | None = None,
        channel_params: ChannelParams | None = None,
    ):
        self.params = sensing_params or SensingParams()
        self.channel = ChannelModel(channel_params)

    # ------------------------------------------------------------------
    # Beampattern
    # ------------------------------------------------------------------

    def steering_vector(
        self,
        uav_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Simplified ULA steering vector toward target.

        Parameters
        ----------
        uav_pos : (3,) — single UAV position
        target_pos : (3,) — single target position

        Returns
        -------
        a : (n_antennas,) complex steering vector
        """
        diff = target_pos - uav_pos
        azimuth = np.arctan2(diff[1], diff[0])
        n = self.params.n_antennas
        d_over_lambda = 0.5  # half-wavelength spacing
        indices = np.arange(n)
        phase = 2.0 * np.pi * d_over_lambda * indices * np.sin(azimuth)
        return np.exp(1j * phase) / np.sqrt(n)

    def beampattern_gain_db(
        self,
        uav_pos: np.ndarray,
        target_pos: np.ndarray,
        beam_direction: np.ndarray | None = None,
    ) -> float:
        """
        Beampattern gain [dB] toward a target.

        If beam_direction is None, assumes beam is steered directly
        at the target (maximum gain = N antennas = 10*log10(N)).
        """
        if beam_direction is None:
            # Perfect steering → gain = N
            return 10.0 * np.log10(self.params.n_antennas)

        a_target = self.steering_vector(uav_pos, target_pos)
        a_beam = self.steering_vector(uav_pos, beam_direction)
        gain = np.abs(np.dot(a_beam.conj(), a_target)) ** 2
        return 10.0 * np.log10(np.maximum(gain, 1e-10))

    # ------------------------------------------------------------------
    # Sensing SNR (two-way / radar equation)
    # ------------------------------------------------------------------

    def sensing_snr_db(
        self,
        uav_pos: np.ndarray,
        target_pos: np.ndarray,
        total_power_dbm: float = 30.0,
    ) -> np.ndarray:
        """
        Sensing SNR [dB] for target detection.

        Uses simplified radar equation:
            SNR_s = (P_s * G^2 * lambda^2 * sigma) /
                    ((4*pi)^3 * d^4 * N_0)

        Parameters
        ----------
        uav_pos : (N, 3) or (3,)
        target_pos : (Q, 3) or (3,)
        total_power_dbm : float — total UAV transmit power [dBm]

        Returns
        -------
        snr : (N, Q) sensing SNR [dB]
        """
        uav = np.atleast_2d(uav_pos)
        tgt = np.atleast_2d(target_pos)

        d = self.channel.distance_3d(uav, tgt)  # (N, Q)
        d = np.maximum(d, 1.0)

        # Sensing power
        p_s_dbm = total_power_dbm + 10.0 * np.log10(
            self.params.sensing_power_ratio
        )

        # Antenna gain (assuming perfect steering) — both TX and RX
        g_db = 10.0 * np.log10(self.params.n_antennas)

        # Radar equation in dB
        lam = self.channel._wavelength
        rcs_db = 10.0 * np.log10(self.params.radar_cross_section)
        lam_db = 20.0 * np.log10(lam)
        const_db = 30.0 * np.log10(4.0 * np.pi)  # (4*pi)^3
        d4_db = 40.0 * np.log10(d)

        # Coherent integration gain
        n_pulses_db = 10.0 * np.log10(self.params.n_pulses)

        snr = (
            p_s_dbm
            + 2 * g_db
            + lam_db
            + rcs_db
            + n_pulses_db
            - const_db
            - d4_db
            - self.params.noise_power_dbm
        )
        return snr

    def sensing_snr_linear(
        self,
        uav_pos: np.ndarray,
        target_pos: np.ndarray,
        total_power_dbm: float = 30.0,
    ) -> np.ndarray:
        """Sensing SNR (linear scale)."""
        return 10.0 ** (
            self.sensing_snr_db(uav_pos, target_pos, total_power_dbm) / 10.0
        )

    # ------------------------------------------------------------------
    # Cramér-Rao Bound for range estimation
    # ------------------------------------------------------------------

    def crb_range(
        self,
        uav_pos: np.ndarray,
        target_pos: np.ndarray,
        total_power_dbm: float = 30.0,
    ) -> np.ndarray:
        """
        CRB for range estimation [m²].

        CRB(d) = c² / (8 * pi² * beta² * SNR_s)

        where beta is the effective bandwidth.

        Returns
        -------
        crb : (N, Q) — CRB in m²  (take sqrt for RMSE bound)
        """
        snr = self.sensing_snr_linear(uav_pos, target_pos, total_power_dbm)
        snr = np.maximum(snr, 1e-10)
        beta = self.channel.params.bandwidth_mhz * 1e6  # effective BW [Hz]
        c = self.SPEED_OF_LIGHT
        crb = c ** 2 / (8.0 * np.pi ** 2 * beta ** 2 * snr)
        return crb

    def crb_range_rmse(
        self,
        uav_pos: np.ndarray,
        target_pos: np.ndarray,
        total_power_dbm: float = 30.0,
    ) -> np.ndarray:
        """sqrt(CRB) — lower bound on range RMSE [m]."""
        return np.sqrt(self.crb_range(uav_pos, target_pos, total_power_dbm))
