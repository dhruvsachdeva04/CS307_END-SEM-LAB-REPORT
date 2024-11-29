import numpy as np

class HopfieldTSPSolver:
    def __init__(self, num_cities, distance_matrix):
        self.num_cities = num_cities
        self.distance_matrix = distance_matrix
        self.num_neurons = num_cities * num_cities
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        self.biases = np.zeros(self.num_neurons)
    
    def setup_weights(self, A=500, B=500, C=1):
        """
        Define weights and biases based on the energy function.
        """
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                current_neuron = i * self.num_cities + j

                # Constraint: Each city must appear exactly once in the tour
                for k in range(self.num_cities):
                    if k != j:
                        neighbor = i * self.num_cities + k
                        self.weights[current_neuron, neighbor] -= A

                # Constraint: Each position must be occupied by exactly one city
                for k in range(self.num_cities):
                    if k != i:
                        neighbor = k * self.num_cities + j
                        self.weights[current_neuron, neighbor] -= B

                # Distance penalty: Tour length minimization
                for k in range(self.num_cities):
                    if i != k:
                        next_neuron = k * self.num_cities + (j + 1) % self.num_cities
                        self.weights[current_neuron, next_neuron] -= C * self.distance_matrix[i][k]

    def energy(self, state):
        """
        Compute the energy of the current state.
        """
        state_matrix = state.reshape((self.num_cities, self.num_cities))
        row_violations = np.sum((np.sum(state_matrix, axis=1) - 1) ** 2)
        col_violations = np.sum((np.sum(state_matrix, axis=0) - 1) ** 2)
        tour_length = 0
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if state_matrix[i, j] == 1:
                    next_city = np.argmax(state_matrix[:, (j + 1) % self.num_cities])
                    tour_length += self.distance_matrix[i][next_city]
        return A * row_violations + B * col_violations + C * tour_length

    def solve(self, max_iterations=100):
        """
        Solve the TSP using the Hopfield network.
        """
        state = np.random.choice([0, 1], self.num_neurons)  # Random initial state
        state = self.correct_state(state)  # Ensure valid initial state
        for _ in range(max_iterations):
            neuron = np.random.randint(0, self.num_neurons)
            # Update neuron state based on weighted sum
            input_sum = np.dot(self.weights[neuron], state) + self.biases[neuron]
            state[neuron] = 1 if input_sum > 0 else 0
            state = self.correct_state(state)
            if self.energy(state) == 0:
                break
        return state.reshape((self.num_cities, self.num_cities))

    def correct_state(self, state):
        """
        Ensure that each row and column has exactly one active neuron.
        """
        state_matrix = state.reshape((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            if np.sum(state_matrix[i]) > 1:
                state_matrix[i, :] = 0
                state_matrix[i, np.random.randint(0, self.num_cities)] = 1
        for j in range(self.num_cities):
            if np.sum(state_matrix[:, j]) > 1:
                state_matrix[:, j] = 0
                state_matrix[np.random.randint(0, self.num_cities), j] = 1
        return state_matrix.flatten()

# Example usage
num_cities = 10
distance_matrix = np.random.randint(1, 100, size=(num_cities, num_cities))
np.fill_diagonal(distance_matrix, 0)  # No self-loops

solver = HopfieldTSPSolver(num_cities, distance_matrix)
solver.setup_weights(A=500, B=500, C=1)

solution = solver.solve()

print("Solved TSP Tour:")
print(solution)