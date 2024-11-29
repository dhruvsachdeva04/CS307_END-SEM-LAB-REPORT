import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """
        Train the network with given patterns using Hebbian learning.
        """
        for pattern in patterns:
            # Reshape pattern into a vector and ensure binary values are -1, 1
            pattern = np.where(pattern.flatten() == 0, -1, 1)
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)  # No self-connections
        self.weights /= len(patterns)

    def recall(self, pattern, steps=5):
        """
        Recall a pattern by iteratively updating the network states.
        """
        pattern = np.where(pattern.flatten() == 0, -1, 1)
        for _ in range(steps):
            for i in range(self.size):
                # Update each neuron based on the weighted sum of its inputs
                weighted_sum = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if weighted_sum >= 0 else -1
        return pattern.reshape(int(np.sqrt(self.size)), -1)

# Define patterns (10x10 binary patterns)
patterns = [
    np.random.choice([0, 1], (10, 10)),  # Random pattern 1
    np.random.choice([0, 1], (10, 10))  # Random pattern 2
]

# Create the network
hopfield = HopfieldNetwork(size=100)

# Train the network
hopfield.train(patterns)

# Test recall with a noisy pattern
test_pattern = patterns[0].copy()
test_pattern[0, 0] = 1 - test_pattern[0, 0]  # Flip one bit for noise
recalled_pattern = hopfield.recall(test_pattern)

print("Original Pattern:")
print(patterns[0])
print("\nNoisy Input:")
print(test_pattern)
print("\nRecalled Pattern:")
print(recalled_pattern)
