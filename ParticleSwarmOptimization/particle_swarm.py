import random
import math
import matplotlib.pyplot as plt

# User defined parameters
x_bounds = [0, 100]  # Boundaries for x-coordinates
y_bounds = [0, 100]  # Boundaries for y-coordinates
num_warehouses = 3   # Number of warehouses to optimize
num_particles = 500  # Number of particles in the swarm
num_iterations = 1000  # Number of optimization iterations
i_weight = 0.7  # Inertia weight: controls the impact of the previous velocity
m_weight = 1.5  # Memory weight: controls the particle's tendency to return to its best known position
s_weight = 1.5  # Social weight: controls the influence of the global best position

# Generate random store and residential locations
num_stores = 3
num_residential = 13
store_locations = [[random.randint(*x_bounds), random.randint(*y_bounds)] for _ in range(num_stores)]
residential_locations = [[random.randint(*x_bounds), random.randint(*y_bounds)] for _ in range(num_residential)]

# Or use predefined store and residential locations used in the video
# store_locations = [
#     [20,20],
#     [20,80],
#     [80,20],
#     [80,80]
# ]
# residential_locations = [
#     [15,10],
#     [30,90],
#     [10,35],
#     [50,40]
# ]

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def fitness_function(*warehouse_locations):
    """
    Calculate the fitness of a given warehouse configuration.
    
    The fitness is determined by maximizing the minimum distance from residential areas
    while minimizing the maximum distance to stores.
    """
    # Reshape the input into pairs of (x, y) coordinates
    warehouse_locations = zip(warehouse_locations[0::2], warehouse_locations[1::2])
    
    store_distances = []
    residential_distances = []
    
    # Calculate distances from each warehouse to all stores and residential areas
    for warehouse in warehouse_locations:
        store_distances.append([distance(warehouse, store) for store in store_locations])
        residential_distances.append([distance(warehouse, res) for res in residential_locations])
    
    # For each store, find the distance to the nearest warehouse
    min_store_distances = [min(dists) for dists in zip(*store_distances)]
    # For each residential area, find the distance to the nearest warehouse
    min_residential_distances = [min(dists) for dists in zip(*residential_distances)]
    
    # Calculate fitness: maximize min distance from residential while minimizing max distance to stores
    fitness = min(min_residential_distances) - max(min_store_distances)
    return fitness

class Particle:
    """
    Represents a single particle in the swarm.
    Each particle has a position, velocity, and memory of its best position.
    """
    def __init__(self, bounds):
        # Initialize particle with random position and velocity within bounds
        self.position = [random.uniform(bound[0], bound[1]) for bound in bounds]
        self.velocity = [random.uniform(-1, 1) for _ in bounds]
        self.best_position = self.position[:]
        self.best_fitness = fitness_function(*self.position)
        
    def update_velocity(self, global_best_position, i_weight, m_weight, s_weight):
        """Update the particle's velocity based on inertia, memory, and social influence."""
        for x in range(len(self.position)):
            inertia = i_weight * self.velocity[x]
            memory = m_weight * random.random() * (self.best_position[x] - self.position[x])
            social = s_weight * random.random() * (global_best_position[x] - self.position[x])
            self.velocity[x] = inertia + memory + social
    
    def update_position(self, bounds):
        """Update the particle's position based on its velocity, keeping it within bounds."""
        for x in range(len(self.position)):
            self.position[x] += self.velocity[x]
            self.position[x] = max(bounds[x][0], min(self.position[x], bounds[x][1]))
        
        # Evaluate fitness of new position
        fitness = fitness_function(*self.position)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position[:]

class Swarm:
    """
    Represents the entire swarm of particles.
    Manages the global best position and coordinates the optimization process.
    """
    def __init__(self, num_particles, bounds, i_weight=0.7, m_weight=1.5, s_weight=1.5):
        self.i_weight = i_weight
        self.m_weight = m_weight
        self.s_weight = s_weight
        self.bounds = bounds
        self.particles = [Particle(self.bounds) for _ in range(num_particles)]
        self.global_best_position = self.particles[0].position[:]
        self.global_best_fitness = self.particles[0].best_fitness
        
    def optimize(self, iterations):
        """Run the optimization process for a given number of iterations."""
        for _ in range(iterations):
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.i_weight, self.m_weight, self.s_weight)
                particle.update_position(self.bounds)
                # Update global best if necessary
                if particle.best_fitness > self.global_best_fitness:
                    print(f'Fit: {particle.best_fitness}')
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_position = particle.best_position[:]

def plot_results(x_bounds, y_bounds, store_locations, residential_locations, warehouse_locations):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#000000')
    ax.set_xlim(x_bounds[0], x_bounds[1])
    ax.set_ylim(y_bounds[0], y_bounds[1])
    ax.set_aspect('equal', adjustable='box')

    # Set the background color
    ax.set_facecolor('#171520')

    # Set color of all text, labels, and ticks
    ax.tick_params(axis='both', colors='#45405c')
    ax.xaxis.label.set_color('#45405c')
    ax.yaxis.label.set_color('#45405c')
    plt.title("Optimized Warehouse Locations", color='#dfdfdf')

    # Change grid color and style
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, color='#45405c')
    ax.minorticks_on()

    # Helper function to manage legend entries
    handles, labels = [], []

    def add_plot(x, y, marker, color, label, handles, labels):
        if label not in labels:
            labels.append(label)
            line, = ax.plot(x, y, marker=marker, markersize=6, color=color, label=label)
            handles.append(line)
        else:
            ax.plot(x, y, marker=marker, markersize=6, color=color)

    # Plot stores with a specific green color
    for store in store_locations:
        add_plot(store[0], store[1], 's', '#6AFF9B', 'Store', handles, labels)

    # Plot residential areas with a specific red color
    for res in residential_locations:
        add_plot(res[0], res[1], 's', '#f97e72', 'Residential', handles, labels)

    # Plot the final warehouse locations
    for warehouse in warehouse_locations:
        add_plot(warehouse[0], warehouse[1], 'o', '#FFFFFF', 'Warehouse', handles, labels)

    plt.xlabel("X", color='#dfdfdf')
    plt.ylabel("Y", color='#dfdfdf')
    ax.legend(handles=handles, loc='best', facecolor='#171520', edgecolor='#dfdfdf', fontsize='medium', labelcolor='#dfdfdf')
    plt.show()

# Set up the optimization problem
bounds = [x_bounds, y_bounds] * num_warehouses
swarm = Swarm(num_particles, bounds, i_weight, m_weight, s_weight)

# Run the optimization
swarm.optimize(num_iterations)

# Extract the optimized warehouse locations
warehouse_locations = [(swarm.global_best_position[i], swarm.global_best_position[i+1]) for i in range(0, len(swarm.global_best_position), 2)]

# Plot the results
plot_results(x_bounds, y_bounds, store_locations, residential_locations, warehouse_locations)


# Run Command: python particle_swarm.py