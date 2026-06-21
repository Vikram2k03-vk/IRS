import matplotlib.pyplot as plt

from array_factor import generate_radiation_pattern

test_cases = [
    [0, 0, 0, 0],
    [0, 30, 60, 90],
    [0, 60, 120, 180],
    [0, 90, 180, 270]
]

plt.figure(figsize=(10, 6))

for phases in test_cases:

    theta, pattern = generate_radiation_pattern(
        phases
    )

    plt.plot(
        theta,
        pattern,
        label=str(phases)
    )

plt.xlabel("Angle (Degrees)")
plt.ylabel("Normalized Magnitude")
plt.title("Beam Steering Comparison")
plt.legend()
plt.grid(True)

plt.show()