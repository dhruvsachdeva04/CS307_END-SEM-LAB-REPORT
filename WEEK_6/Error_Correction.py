def test_error_correction(network, pattern, max_flips):
    """
    Test the error correction capability of a Hopfield network.
    """
    results = []
    for num_flips in range(max_flips + 1):
        noisy_pattern = pattern.copy()
        flip_indices = np.random.choice(len(pattern), num_flips, replace=False)
        noisy_pattern[flip_indices] *= -1  # Flip bits

        recalled_pattern = network.recall(noisy_pattern)
        is_correct = np.array_equal(recalled_pattern, pattern)
        results.append((num_flips, is_correct))
    return results

# Example usage
num_neurons = 10
patterns = [np.random.choice([-1, 1], size=num_neurons) for _ in range(2)]

network = HopfieldNetwork(num_neurons)
network.train(patterns)

# Test error correction for a single pattern
pattern = patterns[0]
error_correction_results = test_error_correction(network, pattern, max_flips=5)

print("Number of flips vs Correct recall:")
for flips, correct in error_correction_results:
    print(f"{flips} flips: {'Correct' if correct else 'Incorrect'}")