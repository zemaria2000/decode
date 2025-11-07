import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Set the style for the plots and some base configurations
sns.set_style("ticks")
# Set default figure size
plt.rcParams['figure.figsize'] = (10, 6)
# Set text sizes
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.0

# Use LaTeX for rendering text
plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'CMU'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'CMU'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def compute_tree_layout(parents: list, level_spacing: int = 5, horiz_spacing: int = 4):
    """
        Computing the coordinates for each tree node.
        Args:
            - parents (list): list with the parent node identifiers
            - level_spacing (int): parameter for plotting/scaling purposes
            - horiz_spacing (int): parameter for plotting/scaling purposes
        Returns:
            - coords (dict): dictionary with a node identifier and a tuple indicating the data coords in the target plot
    """

    children = defaultdict(list)
    for i, p in enumerate(parents):
        if p != -1:
            children[p].append(i)

    coords, current_x = {}, 0

    def assign_coords(node, depth=0):
        nonlocal current_x
        if node not in children or len(children[node]) == 0:
            coords[node] = (current_x, -depth * level_spacing)
            current_x += horiz_spacing
            return coords[node][0]
        xs = [assign_coords(c, depth + 1) for c in children[node]]
        x_center = (min(xs) + max(xs)) / 2
        coords[node] = (x_center, -depth * level_spacing)
        return x_center

    root = [i for i, p in enumerate(parents) if p == -1][0]
    assign_coords(root)
    return coords


def draw_tree_nodes_and_edges(ax, coords: dict, parent_list: list, labels: list, face_colors: list, text_colors: list):
    """
        Drawing each node box
        Args:
            - ax: figure axis object
            - coords (dict): dictionary with a node identifier and a tuple indicating the data coords in the target plot
            - parent_list (list): list with all the parent nodes
            - labels (list): list with the texts to be input to each node box
            - face_colors (list): list with all the colors for the nodes
            - text_colors (list): list with the text colors for each node
    """

    for i in range(len(parent_list)):
        x, y = coords[i]
        # Node
        ax.text(
            x, y, labels[i], ha="center", va="center",
            color=text_colors[i],
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=face_colors[i], edgecolor="black")
        )
        # Edge
        if parent_list[i] != -1:
            px, py = coords[parent_list[i]]
            ax.plot([px, x], [py, y], 'k-', linewidth=1)


def plot_classification_tree(tree, level_spacing: int = 5, hor_spacing: int = 3, palette = None):
    """
        Plotting a classification tree
        Args:
            - tree: tree object from the base_tree file class
            - palette (list): color palette to be used for class coloring
            - level_spacing (int): parameter for plotting/scaling purposes
            - horiz_spacing (int): parameter for plotting/scaling purposes
    """

    # Auxiliary function to compute whether we have a dark color or not
    def _is_dark_color(color):
        """Determine if a given RGB color is 'dark' based on perceived brightness."""
        r, g, b = color  # Ignore alpha channel
        brightness = 0.299 * r + 0.587 * g + 0.114 * b  # Standard luminance formula
        return brightness < 0.5  # Threshold: lower values are "dark"
    
    # Defining all class colors + attributing all classes to a particular color
    if palette is None:  # Standard class coloring if no palette is provided
        palette = plt.get_cmap('Paired')
        dark_indices = [1, 3, 5, 7, 9, 11]
        dark_colors = [palette(i / palette.N) for i in dark_indices]
        class_colors = []
        for i, _ in enumerate(tree.target_names):
            class_colors.append(dark_colors[i % len(dark_colors)])
        [palette.colors[2*i] if 2*i < palette.N else palette.colors[2*i - palette.N] for i, _ in enumerate(tree.target_names)]
    else:
        class_colors = palette[:len(tree.target_names)]

    # Retrieve the coordinates
    coords = compute_tree_layout(tree.PARENT)

    # Creating all text objects/labels
    face_colors, text_colors, labels = [], [], []
    
    for i in range(len(tree.N_NODE_SAMPLES)):
        r,g,b = to_rgb(class_colors[tree.CLASS[i] % len(class_colors)])
        f = tree.IMPURITY[i]
        blended = (f + (1-f)*r, f + (1-f)*g, f + (1-f)*b)
        face_colors.append(blended)
        text_colors.append("white" if _is_dark_color(blended) else "black")

        # Label text
        if tree.FEATURE[i] is None:
            label = f'Node {i}\n'
        else:
            label = f"Node {i}\n{tree.feature_names[tree.FEATURE[i]]} $\\leq$ {tree.VALUE[i]:.2f}\n"

        label += f"samples={tree.N_NODE_SAMPLES[i]}\nclass={tree.target_names[tree.CLASS[i]]}"
        label += f"\nimpurity={tree.IMPURITY[i]:.2f}\nclass_dist={tree.CLASS_DIST[i]}"
        
        labels.append(label)

    # Figure size configs
    depths = max(tree.DEPTH)+1
    fig_height = depths * (level_spacing/2)
    x_values = [x for x, _ in coords.values()]
    fig_width = (max(x_values) - min(x_values)) * 0.9 # Configurable parameters    

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    draw_tree_nodes_and_edges(ax, coords, tree.PARENT, labels, face_colors, text_colors)
    ax.axis("off")
    plt.tight_layout() 



def plot_regression_tree(tree, box_plots_list: list = None, level_spacing: int = 5, hor_spacing: int = 3):
    """
        Plotting a classification tree
        Args:
            - tree: tree object from the base_tree file class
            - level_spacing (int): parameter for plotting/scaling purposes
            - horiz_spacing (int): parameter for plotting/scaling purposes
            - box_plots_list (list): list with a series of box plots to be plot
    """

    # Auxiliary function to compute whether we have a dark color or not
    def _is_dark_color(color):
        """Determine if a given RGB color is 'dark' based on perceived brightness."""
        r, g, b = color  # Ignore alpha channel
        brightness = 0.299 * r + 0.587 * g + 0.114 * b  # Standard luminance formula
        return brightness < 0.5  # Threshold: lower values are "dark"

    # Auxiliary function for coordinate conversion
    def _pixel_to_data(ax, width_px, height_px):
        ## width
        p1_disp = np.array([0, 0])
        p2_disp = np.array([width_px, 0])
        p1_data, p2_data = ax.transData.inverted().transform([p1_disp, p2_disp])
        width = p2_data[0] - p1_data[0]
        ## height
        p3_disp = np.array([0, 0])
        p4_disp = np.array([0, height_px])
        p3_data, p4_data = ax.transData.inverted().transform([p3_disp, p4_disp])
        height = np.abs(p4_data[1] - p3_data[1])

        return width, height
    
    # Defining manually that the color palette for regression trees is the 'Blues' heatmap from seaborn
    cmap = sns.color_palette("Blues_r", as_cmap = True)

    # Retrieve the coordinates + additional auxiliary variables
    coords = compute_tree_layout(tree.PARENT)
    avg_values = np.array(tree.AVG_VALUE, float).squeeze() # Average values of predictions
    impurities = np.array(tree.IMPURITY, float).squeeze() # Average impurity values
    norm = Normalize(vmin = 0, vmax = impurities.max()) # Normalized values for coloring purposes

    # Retrieving the coordinates for the target boxplots
    ## leaf coordinates
    leaf_coords = {}
    for node_idx, x_y in coords.items():
        if tree.LEAF[node_idx] == 1:
            leaf_coords[node_idx] = x_y
    ordered_leaf_coords = dict(sorted(leaf_coords.items()))
    # Defining the box plot coordinates
    box_plot_coords = {}
    for node_idx, x_y in ordered_leaf_coords.items():
        box_plot_coords[node_idx] = (x_y[0], x_y[1] - 0.33*level_spacing)

    # Creating all text objects/labels
    face_colors, text_colors, labels = [], [], []
    
    for i in range(len(tree.N_NODE_SAMPLES)):
        color = to_rgb(cmap(norm(impurities[i])))[:3]
        face_colors.append(color)
        text_colors.append("white" if _is_dark_color(color) else "black")

        # Label text
        if tree.FEATURE[i] is None:
            label = f'Node {i}\n'
        else:
            label = f"Node {i}\n{tree.feature_names[tree.FEATURE[i]]} $\\leq$ {tree.VALUE[i]:.2f}\n"
        label += f"samples={tree.N_NODE_SAMPLES[i]}\nimpurity={tree.IMPURITY[i]:.2f}\ntarget value={np.round(avg_values[i], 3)}"
       
        labels.append(label)

    # Figure size configs
    depths = max(tree.DEPTH)+1
    fig_height = depths * (level_spacing/2)
    x_values = [x for x, _ in coords.values()]
    fig_width = (max(x_values) - min(x_values)) * 0.7 # Configurable parameters    

    # Creating the main figure for the tree
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # For the regression tree, for plotting purposes, I will need to build the plot elements one by one
    for i in range(len(tree.PARENT)):
        
        # Get coords and label text
        x, y = coords[i]
        label = labels[i]
        
        # Defining the text object
        txt = ax.text(
            x, y, label, ha="center", va="center",
            color=text_colors[i],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=face_colors[i], edgecolor="black")
        )

        # Draw connecting edge
        if i > 0:
            parent = tree.PARENT[i]
            px, py = coords[parent]
            ax.plot([px, x], [py, y], "k-", linewidth=1)

        # --- Add boxplot below leaf node ---
        if tree.LEAF[i] == 1 and box_plots_list is not None:
            fig_box = box_plots_list[i] if i < len(box_plots_list) else None
            if fig_box is None:
                continue
            src_ax = fig_box.axes[0]  # get Axes from the small figure

            # Retrieving the width of the text box
            bbox_patch = txt.get_bbox_patch()
            bbox = bbox_patch.get_bbox()
            width_px, height_px = bbox.width, bbox.height  # NOTE: IN PIXELS!!
            # Converting pixels to data coordinates
            width, height = _pixel_to_data(ax, width_px, height_px)

            # Retrieving the width and height of the boxplot
            width_box_px = src_ax.bbox.intervalx[1] - src_ax.bbox.intervalx[0]
            height_box_px = src_ax.bbox.intervaly[1] - src_ax.bbox.intervaly[0]
            # NOTE: also in pixels, so the same transformation must be made
            # width_box, height_box =  _pixel_to_data(ax, width_box_px, height_box_px)

            # Defining a box scaling factor, to fit within a box with the same width as the target box
            scaling_factor = width_px/width_box_px

            # Defining the coordinates for the box plot
            width_target_box = width_px
            height_target_box = height_box_px * scaling_factor

            # Create inset axis at leaf position
            inset = inset_axes(
                ax,
                width = width_target_box, height = height_target_box,
                bbox_to_anchor=(box_plot_coords[i][0] - 0.6*width_target_box, box_plot_coords[i][1]-height_target_box, width_target_box*hor_spacing/3, height_target_box*0.5),
                bbox_transform=ax.transData,
                loc="center",
                borderpad=1
            )

            # Copy graphical elements (lines, collections) from source Axes
            for line in src_ax.lines:
                inset.plot(
                    line.get_xdata(), line.get_ydata(),
                    color=line.get_color(), linewidth=line.get_linewidth()
                )
            for col in src_ax.collections:
                new_col = col.__class__(col.get_paths())
                new_col.set_offsets(col.get_offsets())
                new_col.set_facecolor(col.get_facecolor())
                new_col.set_edgecolor(col.get_edgecolor())
                new_col.set_sizes(col.get_sizes())
                new_col.set_linewidths(col.get_linewidths())
                inset.add_collection(new_col)

            # inset.axis("off")
            inset.set_yticks([])

    ax.axis("off")

def box_plots(tree, only_leaf: bool = True):
    """
        Plotting box plots for the different nodes in the tree - only useful for regression trees!!
        Args:
            - tree: tree object 
            - only_leaf (bool): if True, only leaf box plots are returned, with the other as None
        Returns:
            - box_plots (list): a list of figures, with information on each box plot
    """

    # Characteristics for the box plots
    boxprops = dict(linewidth=1, color='black')     # Center box characteristics
    whiskerprops = dict(linewidth=1, color='black') # Line characteristics
    capprops = dict(linewidth=1, color='black')     # Top and down lines
    medianprops = dict(linewidth=1.5, color='red')
    meanprops = dict(marker='D', markerfacecolor='red', markeredgecolor='black', markersize=6)
    flierprops = dict(marker='o', markersize=6, markerfacecolor='white', markeredgecolor='red', alpha=0.9)

    box_plots = []

    # List of datasets for each node
    node_ds_list = []
    for ds_idx in tree.NODE_DS:
        node_ds_list.append(tree.normalised_dataset[ds_idx])

    for i, ds in enumerate(node_ds_list):
        
        if only_leaf:
            if tree.LEAF[i] == 0:
                box_plots.append(None)
                continue
        
        # 1. Making predictions on the dataset for the node
        preds = tree.model.predict(ds)
        output = tree._inverse_scale_outputs(preds)
        # Creating now the box plot figure
        fig, ax = plt.subplots(figsize = (4, 2))
        ax.boxplot(output, orientation = 'horizontal', widths = 0.7, boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops, medianprops = medianprops, meanprops = meanprops, flierprops = flierprops, showmeans = True)
        box_plots.append(fig)
        plt.close(fig)

    return box_plots