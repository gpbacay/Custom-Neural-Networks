import matplotlib.pyplot as plt

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