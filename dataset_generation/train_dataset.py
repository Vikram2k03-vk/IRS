import numpy as np
import pandas as pd

from array_factor import generate_radiation_pattern


def phase_to_sincos(phase_deg):

    phase_rad = np.deg2rad(phase_deg)

    return np.cos(phase_rad), np.sin(phase_rad)


def generate_dataset():

    phases = np.arange(0, 361, 20)

    rows = []

    count = 0

    for p2 in phases:
        for p3 in phases:
            for p4 in phases:

                phase_vector = [
                    0,
                    p2,
                    p3,
                    p4
                ]

                _, pattern = generate_radiation_pattern(
                    phase_vector
                )

                row = list(pattern)

                targets = []

                for phase in phase_vector:

                    c, s = phase_to_sincos(
                        phase
                    )

                    targets.extend([
                        c,
                        s
                    ])

                row.extend(
                    targets
                )

                rows.append(
                    row
                )

                count += 1

                if count % 500 == 0:

                    print(
                        f"Generated {count} patterns..."
                    )

    columns = [
        f"P{i}"
        for i in range(181)
    ]

    columns.extend([
        "Cos1","Sin1",
        "Cos2","Sin2",
        "Cos3","Sin3",
        "Cos4","Sin4"
    ])

    df = pd.DataFrame(
        rows,
        columns=columns
    )

    return df


if __name__ == "__main__":

    print(
        "Generating Dataset..."
    )

    df = generate_dataset()

    print(
        "Shape:",
        df.shape
    )

    df.to_csv(
        "../data/train_dataset.csv",
        index=False
    )

    print(
        "Dataset Saved"
    )