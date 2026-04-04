"""
Generate a realistic synthetic CIC-IDS-2017 dataset for testing.

Creates data matching the CIC-IDS-2017 feature schema with realistic
feature distributions and multiple attack classes for pipeline validation.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("xai_ids.generate_sample")

# The 78 network flow features from CIC-IDS-2017 (after removing ID columns)
CIC_IDS_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

# Attack classes in CIC-IDS-2017
ATTACK_CLASSES = {
    "BENIGN": 0.60,
    "DoS Hulk": 0.10,
    "PortScan": 0.08,
    "DDoS": 0.07,
    "DoS GoldenEye": 0.04,
    "FTP-Patator": 0.03,
    "SSH-Patator": 0.03,
    "DoS slowloris": 0.02,
    "DoS Slowhttptest": 0.015,
    "Bot": 0.01,
    "Web Attack - Brute Force": 0.005,
    "Web Attack - XSS": 0.005,
    "Infiltration": 0.002,
    "Web Attack - Sql Injection": 0.002,
    "Heartbleed": 0.001,
}


def _generate_feature_data(
    n_samples: int, label: str, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate synthetic feature values based on the traffic class.

    Different attack types have distinct feature distributions to make
    the classification problem realistic and learnable.
    """
    data = np.zeros((n_samples, len(CIC_IDS_FEATURES)))

    if label == "BENIGN":
        # Normal traffic: moderate values, low variance
        data[:, 0] = rng.choice(
            [80, 443, 8080, 3389, 53], size=n_samples
        )  # Destination Port
        data[:, 1] = rng.exponential(50000, n_samples)  # Flow Duration
        data[:, 2] = rng.poisson(5, n_samples)  # Total Fwd Packets
        data[:, 3] = rng.poisson(4, n_samples)  # Total Backward Packets
        data[:, 4] = rng.exponential(500, n_samples)  # Total Length Fwd
        data[:, 5] = rng.exponential(1000, n_samples)  # Total Length Bwd
        for i in range(6, len(CIC_IDS_FEATURES)):
            data[:, i] = rng.exponential(10, n_samples)
    elif "DoS" in label or "DDoS" in label:
        # DoS: high packet rates, many fwd packets
        data[:, 0] = rng.choice([80, 443], size=n_samples)
        data[:, 1] = rng.exponential(5000, n_samples)
        data[:, 2] = rng.poisson(100, n_samples)  # Many fwd packets
        data[:, 3] = rng.poisson(2, n_samples)  # Few bwd packets
        data[:, 4] = rng.exponential(5000, n_samples)  # High fwd length
        data[:, 5] = rng.exponential(100, n_samples)  # Low bwd length
        for i in range(6, len(CIC_IDS_FEATURES)):
            data[:, i] = rng.exponential(50, n_samples) + 20
    elif "PortScan" in label:
        # Port scan: many different destination ports
        data[:, 0] = rng.integers(1, 65535, n_samples)  # Random ports
        data[:, 1] = rng.exponential(1000, n_samples)  # Short flows
        data[:, 2] = rng.poisson(2, n_samples)
        data[:, 3] = rng.poisson(1, n_samples)
        data[:, 4] = rng.exponential(100, n_samples)
        data[:, 5] = rng.exponential(50, n_samples)
        for i in range(6, len(CIC_IDS_FEATURES)):
            data[:, i] = rng.exponential(5, n_samples)
    elif "Patator" in label:
        # Brute force: many connection attempts
        port = 21 if "FTP" in label else 22
        data[:, 0] = port
        data[:, 1] = rng.exponential(10000, n_samples)
        data[:, 2] = rng.poisson(10, n_samples)
        data[:, 3] = rng.poisson(8, n_samples)
        data[:, 4] = rng.exponential(200, n_samples)
        data[:, 5] = rng.exponential(300, n_samples)
        for i in range(6, len(CIC_IDS_FEATURES)):
            data[:, i] = rng.exponential(15, n_samples) + 5
    elif "Bot" in label:
        # Bot: periodic, automated traffic
        data[:, 0] = rng.choice([80, 443, 8080, 6667], size=n_samples)
        data[:, 1] = rng.exponential(100000, n_samples)
        data[:, 2] = rng.poisson(20, n_samples)
        data[:, 3] = rng.poisson(15, n_samples)
        data[:, 4] = rng.exponential(1000, n_samples)
        data[:, 5] = rng.exponential(800, n_samples)
        for i in range(6, len(CIC_IDS_FEATURES)):
            data[:, i] = rng.exponential(25, n_samples) + 10
    elif "Web Attack" in label:
        # Web attacks: HTTP traffic with unusual patterns
        data[:, 0] = rng.choice([80, 443, 8080], size=n_samples)
        data[:, 1] = rng.exponential(30000, n_samples)
        data[:, 2] = rng.poisson(15, n_samples)
        data[:, 3] = rng.poisson(10, n_samples)
        data[:, 4] = rng.exponential(2000, n_samples)  # Larger payloads
        data[:, 5] = rng.exponential(3000, n_samples)
        for i in range(6, len(CIC_IDS_FEATURES)):
            data[:, i] = rng.exponential(30, n_samples) + 15
    else:
        # Other attacks: general anomalous traffic
        data[:, 0] = rng.integers(1, 65535, n_samples)
        data[:, 1] = rng.exponential(20000, n_samples)
        data[:, 2] = rng.poisson(8, n_samples)
        data[:, 3] = rng.poisson(6, n_samples)
        data[:, 4] = rng.exponential(800, n_samples)
        data[:, 5] = rng.exponential(600, n_samples)
        for i in range(6, len(CIC_IDS_FEATURES)):
            data[:, i] = rng.exponential(20, n_samples) + 8

    # Ensure non-negative values
    data = np.abs(data)

    return data


def generate_sample_dataset(
    n_samples: int = 50000,
    output_dir: str = "data/raw",
    seed: int = 42,
) -> str:
    """
    Generate a synthetic CIC-IDS-2017-format dataset.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate.
    output_dir : str
        Directory to save the CSV file.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    str
        Path to the generated CSV file.
    """
    logger.info(f"Generating synthetic dataset with {n_samples} samples")
    rng = np.random.default_rng(seed)

    all_data = []
    all_labels = []

    for attack_class, proportion in ATTACK_CLASSES.items():
        n = max(int(n_samples * proportion), 200)
        feature_data = _generate_feature_data(n, attack_class, rng)
        labels = [attack_class] * n
        all_data.append(feature_data)
        all_labels.extend(labels)
        logger.info(f"  Generated {n} samples for '{attack_class}'")

    X = np.vstack(all_data)
    df = pd.DataFrame(X, columns=CIC_IDS_FEATURES)
    df["Label"] = all_labels

    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, "sample_cicids2017.csv")
    df.to_csv(output_path, index=False)

    logger.info(f"Saved synthetic dataset: {output_path} ({len(df)} rows)")
    return output_path


if __name__ == "__main__":
    generate_sample_dataset()
