import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))
    
    def train(self, patterns):
        """
        Train the Hopfield network using Hebbian learning.
        """
        num_patterns = len(patterns)
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)  # No self-connections
        self.weights /= num_patterns  # Normalize weights
    
    def recall(self, pattern, max_steps=10):
        """
        Recall a pattern using synchronous updates.
        """
        state = pattern.copy()
        for _ in range(max_steps):
            state = np.sign(np.dot(self.weights, state))
        return state
    
    def test_capacity(self, patterns, noise_level=0.1):
        """
        Test the capacity of the Hopfield network by adding noise to patterns
        and checking recall accuracy.
        """
        successes = 0
        for pattern in patterns:
            # Add noise to the pattern
            noisy_pattern = pattern.copy()
            num_flips = int(noise_level * len(pattern))
            flip_indices = np.random.choice(len(pattern), num_flips, replace=False)
            noisy_pattern[flip_indices] *= -1

            # Recall the pattern
            recalled_pattern = self.recall(noisy_pattern)
            if np.array_equal(recalled_pattern, pattern):
                successes += 1
        return successes / len(patterns)

# Example usage
num_neurons = 100  # Number of neurons
patterns_to_test = 30  # Max number of patterns to test

# Generate random patterns
patterns = [np.random.choice([-1, 1], size=num_neurons) for _ in range(patterns_to_test)]

# Initialize Hopfield network
network = HopfieldNetwork(num_neurons)

# Find actual capacity
proved_capacity = 0
for p in range(1, patterns_to_test + 1):
    network.train(patterns[:p])  # Train on first `p` patterns
    success_rate = network.test_capacity(patterns[:p], noise_level=0.1)
    if success_rate < 1.0:  # Break if any pattern is not recalled correctly
        break
    proved_capacity = p

print(f"Proved capacity: {proved_capacity} patterns")


num_trials = 15
avg_capacity = 0

for _ in range(num_trials):
    # Generate random patterns
    patterns = [np.random.choice([-1, 1], size=10) for _ in range(20)]
    
    # Initialize Hopfield network
    network = HopfieldNetwork(10)

    # Determine capacity for this trial
    proved_capacity = 0
    for p in range(1, 21):  # Test up to 20 patterns
        network.train(patterns[:p])  # Train on the first `p` patterns
        success_rate = network.test_capacity(patterns[:p], noise_level=0.1)
        if success_rate < 1.0:
            break
        proved_capacity = p

    avg_capacity += proved_capacity

avg_capacity /= num_trials
print(f"Average proved capacity over {num_trials} trials: {avg_capacity:.2f} patterns")