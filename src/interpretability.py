from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
from captum.attr import IntegratedGradients


def plot_importance(values, rows, cols, title, grid_labels=True, traffic_light=True):
    """
    General function to plot importance or attribution values.

    Parameters:
        values (np.ndarray): Importance or attribution values.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        title (str): Title for the plot.
        grid_labels (bool): Whether to annotate grid cells with values.
        traffic_light (bool): Whether to plot traffic light circles.

    Returns:
        None
    """
    if traffic_light:
        traffic_light_values = values[:3]
        grid_values = values[3:].reshape(rows, cols)
    else:
        grid_values = values.reshape(rows, cols)

    # Set up the plot with optional traffic light subplot
    if traffic_light:
        fig, (ax, traffic_light_ax) = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": [4, 1]}, figsize=(10, 6)
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot grid values as a heatmap
    cmap = plt.cm.viridis
    im = ax.imshow(grid_values, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, label="Value")
    ax.set_title(title)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")

    # Add text annotations for each grid cell
    if grid_labels:
        for i in range(rows):
            for j in range(cols):
                ax.text(
                    j,
                    i,
                    f"{grid_values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=(
                        "black"
                        if grid_values[i, j] > grid_values.max() / 2
                        else "white"
                    ),
                    fontsize=8,
                )

    # Plot traffic light importance as circles if required
    if traffic_light:
        traffic_light_colors = ["red", "yellow", "green"]

        for idx, color in enumerate(traffic_light_colors):
            circle = patches.Circle(
                (0.5, 2.5 - idx), 0.4, edgecolor="black", facecolor=color
            )
            traffic_light_ax.add_patch(circle)
            traffic_light_ax.text(
                0.5,
                2.5 - idx,
                f"{traffic_light_values[idx]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

        # Adjust traffic light subplot
        traffic_light_ax.set_xlim(0, 1)
        traffic_light_ax.set_ylim(0, 3)
        traffic_light_ax.axis("off")  # Hide axes for clarity
        traffic_light_ax.set_title("Traffic Light Importance")

    plt.tight_layout()
    plt.show()


def plot_input_layer_weights(
    model,
    rows,
    cols,
    input_layer_name="network.0.weight",
    title="Input Layer Weight Heatmap",
):
    for name, param in model.named_parameters():
        if input_layer_name in name:
            weights = param.data
            break

    aggregated_weights = np.abs(weights.numpy()).sum(axis=0)

    plot_importance(aggregated_weights, rows, cols, title)


def plot_weight_attribution(
    model, state_input, rows, cols, title="Attribution Heatmap", wrapper_func=None
):
    """
    Computes and plots the weight attribution for a given state input using Captum.

    Parameters:
        model (torch.nn.Module): The trained model.
        state_input (torch.Tensor): The state input tensor.
        rows (int): Number of rows in the input grid.
        cols (int): Number of columns in the input grid.
        title (str): Title for the plot.

    Returns:
        None
    """
    # Ensure the model is in evaluation mode
    model.eval()

    if wrapper_func:
        ig = IntegratedGradients(wrapper_func)
    else:
        ig = IntegratedGradients(model)

    # Compute attributions
    state_input.requires_grad = True
    attributions, _ = ig.attribute(state_input, target=0, return_convergence_delta=True)

    # Convert attributions to numpy for visualization
    attributions = attributions.detach().numpy().squeeze()

    # Plot the attributions
    plot_importance(attributions, rows, cols, title, grid_labels=True)
