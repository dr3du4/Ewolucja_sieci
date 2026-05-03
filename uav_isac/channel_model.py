"""
Air-to-Ground Channel Model for UAV-ISAC Simulation.

Implements probabilistic LoS/NLoS path loss based on:
- ITU-R / 3GPP Urban Macro model for UAV A2G links
- Free-space path loss (FSPL) as baseline
- Probabilistic LoS model depending on elevation angle

References:
    Al-Hourani et al., "Modeling Air-to-Ground Path Loss for Low Altitude
    Platforms in Urban Environments," IEEE GLOBECOM 2014.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ChannelParams:
    """Channel model parameters."""
    freq_ghz: float = 2.0           # Carrier frequency [GHz]
    env_a: float = 9.61             # Environment param a (urban)
    env_b: float = 0.16             # Environment param b (urban)
    eta_los_db: float = 1.0         # Additional LoS loss [dB]
    eta_nlos_db: float = 20.0       # Additional NLoS loss [dB]
    noise_power_dbm: float = -104.0 # Noise power [dBm] (10 MHz BW)
    tx_power_dbm: float = 30.0      # UAV transmit power [dBm]
    bandwidth_mhz: float = 10.0     # System bandwidth [MHz]


class ChannelModel:
    """
    Probabilistic Air-to-Ground channel model.

    Models the path loss between a UAV at altitude H and a ground node,
    accounting for LoS probability as a function of elevation angle.
    """

    SPEED_OF_LIGHT = 3e8  # m/s

    def __init__(self, params: ChannelParams | None = None):
        self.params = params or ChannelParams()
        self._freq_hz = self.params.freq_ghz * 1e9
        self._wavelength = self.SPEED_OF_LIGHT / self._freq_hz

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def distance_3d(
        self,
        uav_pos: np.ndarray,
        ground_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Euclidean 3D distance between UAV(s) and ground node(s).

        Parameters
        ----------
        uav_pos : (N, 3) or (3,)  — [x, y, z] of UAV(s)
        ground_pos : (K, 3) or (3,) — [x, y, z] of ground nodes (z usually 0)

        Returns
        -------
        dist : (N, K) distance matrix  [m]
        """
        uav = np.atleast_2d(uav_pos)       # (N, 3)
        gnd = np.atleast_2d(ground_pos)     # (K, 3)
        # Broadcasting: (N,1,3) - (1,K,3) -> (N,K,3)
        diff = uav[:, None, :] - gnd[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))  # (N, K)

    def elevation_angle_deg(
        self,
        uav_pos: np.ndarray,
        ground_pos: np.ndarray,
    ) -> np.ndarray:
        """Elevation angle (degrees) from ground node to UAV."""
        uav = np.atleast_2d(uav_pos)
        gnd = np.atleast_2d(ground_pos)
        dz = uav[:, None, 2] - gnd[None, :, 2]          # (N, K)
        dxy = np.sqrt(
            (uav[:, None, 0] - gnd[None, :, 0]) ** 2
            + (uav[:, None, 1] - gnd[None, :, 1]) ** 2
        )
        dxy = np.maximum(dxy, 1e-6)  # avoid division by zero
        return np.degrees(np.arctan2(np.abs(dz), dxy))

    def p_los(
        self,
        uav_pos: np.ndarray,
        ground_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Probability of Line-of-Sight link.

        P_LoS = 1 / (1 + a * exp(-b * (theta - a)))
        where theta is the elevation angle in degrees.
        """
        theta = self.elevation_angle_deg(uav_pos, ground_pos)
        a = self.params.env_a
        b = self.params.env_b
        return 1.0 / (1.0 + a * np.exp(-b * (theta - a)))

    def fspl_db(self, dist_m: np.ndarray) -> np.ndarray:
        """Free-space path loss [dB]."""
        dist_m = np.maximum(dist_m, 1e-6)
        return 20.0 * np.log10(4.0 * np.pi * dist_m / self._wavelength)

    def path_loss_db(
        self,
        uav_pos: np.ndarray,
        ground_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Average path loss [dB] (LoS/NLoS weighted).

        PL = P_LoS * (FSPL + eta_LoS) + P_NLoS * (FSPL + eta_NLoS)
        """
        d = self.distance_3d(uav_pos, ground_pos)
        fspl = self.fspl_db(d)
        p = self.p_los(uav_pos, ground_pos)

        pl_los = fspl + self.params.eta_los_db
        pl_nlos = fspl + self.params.eta_nlos_db

        return p * pl_los + (1.0 - p) * pl_nlos

    # ------------------------------------------------------------------
    # Higher-level metrics
    # ------------------------------------------------------------------

    def received_power_dbm(
        self,
        uav_pos: np.ndarray,
        ground_pos: np.ndarray,
    ) -> np.ndarray:
        """Received power [dBm] at ground node."""
        return self.params.tx_power_dbm - self.path_loss_db(uav_pos, ground_pos)

    def snr_db(
        self,
        uav_pos: np.ndarray,
        ground_pos: np.ndarray,
    ) -> np.ndarray:
        """SNR [dB] at ground node."""
        return self.received_power_dbm(uav_pos, ground_pos) - self.params.noise_power_dbm

    def snr_linear(
        self,
        uav_pos: np.ndarray,
        ground_pos: np.ndarray,
    ) -> np.ndarray:
        """SNR (linear scale)."""
        return 10.0 ** (self.snr_db(uav_pos, ground_pos) / 10.0)

    def capacity_bps(
        self,
        uav_pos: np.ndarray,
        ground_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Shannon capacity [bps] per link.

        C = B * log2(1 + SNR)
        """
        bw = self.params.bandwidth_mhz * 1e6
        snr = self.snr_linear(uav_pos, ground_pos)
        return bw * np.log2(1.0 + snr)
