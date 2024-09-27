import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generic parameters
x_bounds = [0, 100]  # Boundaries for x-coordinates
y_bounds = [0, 100]  # Boundaries for y-coordinates
num_points = 3  # Number of points to optimize
num_particles = 1000  # Number of particles in the swarm
num_iterations = 1000  # Maximum number of optimization iterations
i_weight = 0.7  # Inertia weight: controls the impact of the previous velocity
m_weight = 1.5  # Memory weight: controls the particle's tendency to return to its best-known position
s_weight = 1.5  # Social weight: controls the influence of the global best position

# Generate random locations for entities (these could be stores, obstacles, targets, etc.)
num_targets = 3
num_obstacles = 3
target_locations = [[random.randint(*x_bounds), random.randint(*y_bounds)] for _ in range(num_targets)]
obstacle_locations = [[random.randint(*x_bounds), random.randint(*y_bounds)] for _ in range(num_obstacles)]

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def fitness_function(*positions, maximize_obstacle_distance=True):
    """
    Generic fitness function to evaluate a configuration of points.
    - maximize_obstacle_distance: If True, maximize the minimum distance from obstacles.
                                  If False, minimize the distance to obstacles.
    """
    positions = zip(positions[0::2], positions[1::2])
    
    target_distances = []
    obstacle_distances = []
    
    # Calculate distances from each point to all targets and obstacles
    for position in positions:
        target_distances.append([distance(position, target) for target in target_locations])
        obstacle_distances.append([distance(position, obstacle) for obstacle in obstacle_locations])
    
    # Find the minimum distance to targets and obstacles for each point
    min_target_distances = [min(dists) for dists in zip(*target_distances)]
    min_obstacle_distances = [min(dists) for dists in zip(*obstacle_distances)]
    
    # Calculate fitness based on maximizing or minimizing distance from obstacles
    if maximize_obstacle_distance:
        fitness = min(min_obstacle_distances) - max(min_target_distances)
    else:
        fitness = max(min_target_distances) - min(min_obstacle_distances)
    
    return fitness

class Particle:
    def __init__(self, bounds):
        self.position = [random.uniform(bound[0], bound[1]) for bound in bounds]
        self.velocity = [random.uniform(-1, 1) for _ in bounds]
        self.best_position = self.position[:]
        self.best_fitness = fitness_function(*self.position)
        
    def update_velocity(self, global_best_position, i_weight, m_weight, s_weight):
        for x in range(len(self.position)):
            inertia = i_weight * self.velocity[x]
            memory = m_weight * random.random() * (self.best_position[x] - self.position[x])
            social = s_weight * random.random() * (global_best_position[x] - self.position[x])
            self.velocity[x] = inertia + memory + social
    
    def update_position(self, bounds):
        for x in range(len(self.position)):
            self.position[x] += self.velocity[x]
            self.position[x] = max(bounds[x][0], min(self.position[x], bounds[x][1]))
        
        fitness = fitness_function(*self.position)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position[:]

class Swarm:
    def __init__(self, num_particles, bounds, i_weight=0.7, m_weight=1.5, s_weight=1.5):
        self.i_weight = i_weight
        self.m_weight = m_weight
        self.s_weight = s_weight
        self.bounds = bounds
        self.particles = [Particle(self.bounds) for _ in range(num_particles)]
        self.global_best_position = self.particles[0].position[:]
        self.global_best_fitness = self.particles[0].best_fitness
        
    def optimize(self):
        for particle in self.particles:
            particle.update_velocity(self.global_best_position, self.i_weight, self.m_weight, self.s_weight)
            particle.update_position(self.bounds)
            if particle.best_fitness > self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_position = particle.best_position[:]
        return self.get_all_positions()

    def get_all_positions(self):
        return [particle.position for particle in self.particles]

def main():
    global anim, prev_best_fitness, convergence_count
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#171520')
    ax.set_xlim(x_bounds[0], x_bounds[1])
    ax.set_ylim(y_bounds[0], y_bounds[1])
    ax.set_aspect('equal', adjustable='box')

    ax.tick_params(axis='both', colors='#45405c')
    ax.xaxis.label.set_color('#45405c')
    ax.yaxis.label.set_color('#45405c')
    plt.title("Particle Swarm Optimization", color='#dfdfdf')

    ax.grid(True, which='both', linestyle='--', linewidth=0.7, color='#45405c')
    ax.minorticks_on()

    # Plot target locations
    for target in target_locations:
        ax.plot(target[0], target[1], 's', markersize=6, color='#6AFF9B')

    # Plot obstacle locations
    for obs in obstacle_locations:
        ax.plot(obs[0], obs[1], 's', markersize=6, color='#f97e72')

    # Initialize particles plot
    particles_plot, = ax.plot([], [], 'o', markersize=4, color='#FFFFFF', alpha=0.5)

    # Set up the optimization problem
    bounds = [x_bounds, y_bounds] * num_points
    swarm = Swarm(num_particles, bounds, i_weight, m_weight, s_weight)

    # Initialize previous best fitness
    prev_best_fitness = float('-inf')
    convergence_count = 0
    convergence_threshold = 50  # Number of iterations without improvement to consider convergence

    # Animation update function
    def update(frame):
        global prev_best_fitness, convergence_count
        
        positions = swarm.optimize()
        x_positions = [pos[i] for pos in positions for i in range(0, len(pos), 2)]
        y_positions = [pos[i] for pos in positions for i in range(1, len(pos), 2)]
        particles_plot.set_data(x_positions, y_positions)
        
        # Print current best fitness to terminal
        print(f"Iteration {frame + 1}: Best Fitness = {swarm.global_best_fitness:.4f}")
        
        # Check for convergence
        if swarm.global_best_fitness > prev_best_fitness:
            prev_best_fitness = swarm.global_best_fitness
            convergence_count = 0
        else:
            convergence_count += 1
        
        # Stop if converged
        if convergence_count >= convergence_threshold:
            print(f"Optimal solution found at iteration {frame + 1}")
            anim.event_source.stop()
            plot_final_results()
        
        return particles_plot,

    def plot_final_results():
        # Plot the final optimized locations
        optimized_locations = [(swarm.global_best_position[i], swarm.global_best_position[i + 1]) 
                            for i in range(0, len(swarm.global_best_position), 2)]
        for location in optimized_locations:
            ax.plot(location[0], location[1], 'o', markersize=8, color='#FFFFFF')

        # Update legend
        ax.plot([], [], 's', color='#6AFF9B', label='Target')
        ax.plot([], [], 's', color='#f97e72', label='Obstacle')
        ax.plot([], [], 'o', color='#FFFFFF', label='Optimized Points')

        # Place legend outside the plot area
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), facecolor='#171520', edgecolor='#dfdfdf',
                fontsize='small', labelcolor='#dfdfdf')

        plt.draw()


    # Create the animation
    anim = FuncAnimation(fig, update, frames=num_iterations, interval=50, blit=True)

    plt.xlabel("X", color='#dfdfdf')
    plt.ylabel("Y", color='#dfdfdf')

    plt.show()

# Call the main function when the script is executed
if __name__ == "__main__":
    main()



# Run Command: python particle_swarm_v2.py