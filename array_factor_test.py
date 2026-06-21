from dataset_generation.array_factor import generate_radiation_pattern

theta, pattern = generate_radiation_pattern(
    [0, 0, 0, 0]
)

print("Theta points:", len(theta))
print("Pattern points:", len(pattern))
print("Maximum value:", max(pattern))