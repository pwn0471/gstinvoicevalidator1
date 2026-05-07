import numpy as np
import pandas as pd
from pathlib import Path


def create_invoice_dataset(path: Path, n_samples: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    invoice_amount = rng.uniform(100, 100000, size=n_samples)
    gst_rate = rng.choice([0.0, 5.0, 12.0, 18.0, 28.0], size=n_samples, p=[0.05, 0.25, 0.3, 0.3, 0.1])
    vendor_frequency = rng.integers(1, 50, size=n_samples)
    expected_total = invoice_amount * (1 + gst_rate / 100)
    noise = rng.normal(0, invoice_amount * 0.01, size=n_samples)
    total_amount = expected_total + noise

    deviation = np.abs(total_amount - expected_total)
    suspicious = (
        (deviation > invoice_amount * 0.03)
        | ((gst_rate == 0.0) & (invoice_amount > 50000))
        | ((gst_rate == 28.0) & (invoice_amount < 500.0))
        | (vendor_frequency <= 3)
    )

    labels = np.where(suspicious, "Suspicious", "Valid")
    df = pd.DataFrame(
        {
            "invoice_amount": np.round(invoice_amount, 2),
            "gst_rate": gst_rate,
            "total_amount": np.round(total_amount, 2),
            "vendor_frequency": vendor_frequency,
            "label": labels,
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parent / "dataset.csv"
    df = create_invoice_dataset(dataset_path, n_samples=1000)
    print(f"Saved synthetic dataset to {dataset_path}")
    print(df.head().to_string(index=False))
