from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt


# --- Helper Functions ---

def generate_quadrant_colors():
    """Generates a color map for the 16 hyperquadrants of a 4D space."""
    base_colormaps = {
        0: plt.get_cmap('Oranges'), 1: plt.get_cmap('Blues'),
        2: plt.get_cmap('Greens'), 3: plt.get_cmap('Purples'),
        4: plt.get_cmap('Reds')
    }
    signs = list(product([-1, 1], repeat=4))
    grouped_quadrants = {i: [] for i in range(5)}
    for s in signs:
        num_positives = sum(1 for x in s if x > 0)
        grouped_quadrants[num_positives].append(s)

    quadrant_to_color = {}
    for num_pos, quadrant_list in grouped_quadrants.items():
        colormap = base_colormaps[num_pos]
        num_shades = len(quadrant_list)
        for i, quadrant in enumerate(quadrant_list):
            shade_value = 0.2 + 0.7 * (i / (num_shades - 1)) if num_shades > 1 else 0.6
            quadrant_to_color[quadrant] = colormap(shade_value)

    color_map_array = np.zeros((16, 4))
    for i in range(16):
        binary_rep = format(i, '04b')
        sign_tuple = tuple(1 if bit == '1' else -1 for bit in binary_rep)
        color_map_array[i] = quadrant_to_color[sign_tuple]
    return color_map_array


def gram_schmidt(vectors):
    """Applies the Gram-Schmidt process to make vectors orthonormal."""
    basis = []
    for v in vectors:
        w = v - sum(np.dot(v, b) * b for b in basis)
        if np.linalg.norm(w) > 1e-10:
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)


# --- Main Plotting Function ---

def get_fig(
        fig, ax, p0, v_unortho_1, v_unortho_2,
        resolution=800, u_range=(-8, 8), v_range=(-8, 8),
        boundary_linewidth=2.0, min_region_size=1000,
        add_arrows=True, arrow_step=25, arrow_length=0.3,
        add_hatching=True,
        # --- NEW: RG Flow Parameters ---
        add_flow=False,
        M=None,
        flow_color='k',
        flow_density=1.5,
        flow_linewidth=1.2
):
    """
    Generates and saves a 2D slice of 4D hyperquadrants with advanced features,
    including an optional RG flow vector field.
    """
    print(f"Generating plot...")

    v1, v2 = gram_schmidt([v_unortho_1, v_unortho_2])
    color_map = generate_quadrant_colors()

    u = np.linspace(u_range[0], u_range[1], resolution)
    v = np.linspace(v_range[0], v_range[1], resolution)
    u_grid, v_grid = np.meshgrid(u, v)
    points_4d = p0 + u_grid[..., np.newaxis] * v1 + v_grid[..., np.newaxis] * v2

    signs = np.sign(points_4d + 1e-12)
    quadrant_indices = ((signs + 1) / 2 @ np.array([8, 4, 2, 1])).astype(int)
    image = color_map[quadrant_indices]

    ax.imshow(image, origin='lower', extent=[u_range[0], u_range[1], v_range[0], v_range[1]])
    ax.axis('off')

    # --- NEW: RG FLOW VECTOR FIELD (STREAMPLOT) ---
    if add_flow and M is not None:
        print("Adding RG flow vector field...")
        # 1. Calculate 4D velocity field at each point: vel_4d = M @ p_4d
        # We use np.einsum for efficient batch matrix-vector multiplication over the grid.
        vel_4d = np.einsum('ij,abj->abi', M, points_4d)

        # 2. Project the 4D velocity field onto the 2D plot plane (spanned by v1, v2)
        # The components of the 2D velocity vector are the dot products with the basis vectors.
        vel_u = np.einsum('...i,i->...', vel_4d, v1)
        vel_v = np.einsum('...i,i->...', vel_4d, v2)

        # 3. Plot the projected 2D flow field using streamplot
        ax.streamplot(
            u_grid, v_grid, vel_u, vel_v,
            color=flow_color,
            density=flow_density,
            linewidth=flow_linewidth,
            arrowstyle='->',
            arrowsize=1.5,
            zorder=5  # Ensure flow is visible on top of colors but below boundaries/labels
        )

    # --- HATCHING FOR SPECIAL PHASES (n=0 and n=4) ---
    if add_hatching:
        print("Adding hatching to special phases...")
        index_to_n_map = np.array([bin(i).count('1') for i in range(16)])
        group_id_image = index_to_n_map[quadrant_indices]
        plt.rcParams['hatch.color'] = 'gray'
        plt.rcParams['hatch.linewidth'] = 1.0
        mask_n4 = (group_id_image == 4)
        ax.contourf(u_grid, v_grid, mask_n4, levels=[0.5, 1.5], colors='none', hatches=['//'], zorder=3)
        mask_n0 = (group_id_image == 0)
        ax.contourf(u_grid, v_grid, mask_n0, levels=[0.5, 1.5], colors='none', hatches=['\\\\'], zorder=3)

    print("Drawing vector boundaries and arrows...")
    all_contour_sets = []
    for coord_idx in range(4):
        Z_coord = points_4d[:, :, coord_idx]
        cs = ax.contour(u_grid, v_grid, Z_coord, levels=[0], colors='black', linewidths=boundary_linewidth)
        all_contour_sets.append(cs)

    if add_arrows:
        for cs in all_contour_sets:
            for path in cs.allsegs[0]:
                if len(path) < 2: continue
                for i in range(0, len(path) - 1, arrow_step):
                    start_point, end_point = path[i], path[i + 1]
                    tangent_2d = end_point - start_point
                    u_coord, v_coord = start_point
                    p_4d = p0 + u_coord * v1 + v_coord * v2
                    outward_vector_2d = np.array([np.dot(p_4d, v1), np.dot(p_4d, v2)])
                    if np.dot(tangent_2d, outward_vector_2d) < 0:
                        tangent_2d = -tangent_2d
                    norm = np.linalg.norm(tangent_2d)
                    if norm < 1e-6: continue
                    arrow_vec = (tangent_2d / norm) * arrow_length
                    ax.arrow(start_point[0], start_point[1], arrow_vec[0], arrow_vec[1],
                             head_width=0.15, head_length=0.2, fc='black', ec='black', length_includes_head=True,
                             zorder=10)

    print("Placing smart labels...")
    fig.canvas.draw()
    renderer = fig.canvas.renderer
    index_to_n_map = np.array([bin(i).count('1') for i in range(16)])

    for i in range(16):
        mask = (quadrant_indices == i)
        if np.sum(mask) > min_region_size:
            n = index_to_n_map[i]
            if n == 0:
                name, fs = ' (trivial)', 32
            elif n == 4:
                name, fs = ' ($\\mathbf{E_8}$)', 32
            else:
                name, fs = '', 25
            label_text = "$\\kappa_{\\mathrm{xy}} =" + f"{2 * n}$" + name

            dummy_text = ax.text(0, 0, label_text, fontsize=fs, fontweight='bold')
            bbox_data = dummy_text.get_window_extent(renderer)
            width_px, height_px = bbox_data.width, bbox_data.height
            safe_width, safe_height = height_px / 2.0, width_px / 2.0
            dummy_text.remove()

            dist_map = distance_transform_edt(mask)
            center_x, center_y = center_of_mass(mask)

            safe_factor = 0.95  # weird but better
            # Give a little extra padding
            if center_x < safe_width * safe_factor:
                final_x = safe_width * safe_factor
                print(f'Moving label {label_text} to x={final_x} (was {center_x})')
            elif center_x > resolution - safe_width * safe_factor:
                final_x = resolution - safe_width * safe_factor
                print(f'Moving label {label_text} to x={final_x} (was {center_x})')
            else:
                final_x = center_x
            if center_y < safe_height * safe_factor:
                final_y = safe_height * safe_factor
                print(f'Moving label {label_text} to y={final_y} (was {center_y})')
            elif center_y > resolution - safe_height * safe_factor:
                final_y = resolution - safe_height * safe_factor
                print(f'Moving label {label_text} to y={final_y} (was {center_y})')
            else:
                final_y = center_y
            # --- 4. Convert Pixel Position to Data Coords and Draw ---
            row, col = final_x, final_y
            u_coord = u_range[0] + (col / (resolution - 1)) * (u_range[1] - u_range[0])
            v_coord = v_range[0] + (row / (resolution - 1)) * (v_range[1] - v_range[0])

            txt = ax.text(u_coord, v_coord, label_text, ha='center', va='center', color='white', fontsize=fs,
                          fontweight='bold', zorder=20)
            # txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='black')])
    return fig, ax


def generate_4d_slice_plot(
        p0, v_unortho_1, v_unortho_2, filename,
        resolution=800, u_range=(-8, 8), v_range=(-8, 8),
        boundary_linewidth=2.0, min_region_size=1000,
        add_arrows=True, arrow_step=25, arrow_length=0.3,
        add_hatching=True, show_plot=True,
        # --- NEW: Pass RG Flow parameters ---
        add_flow=False, M=None, flow_color='k',
        flow_density=1.5, flow_linewidth=1.2
):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    fig, ax = get_fig(fig, ax, p0, v_unortho_1, v_unortho_2,
                      resolution, u_range, v_range,
                      boundary_linewidth, min_region_size,
                      add_arrows, arrow_step, arrow_length,
                      add_hatching,
                      # --- NEW: Pass arguments to get_fig ---
                      add_flow=add_flow, M=M, flow_color=flow_color,
                      flow_density=flow_density, flow_linewidth=flow_linewidth)
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()
    plt.close(fig)
    print("Done.")


def main():
    """
    Main script to create the figure with precise manual layout
    and save the final PDF.
    """
    # --- 1. Define Layout Parameters in Inches ---
    ax_width_in = 10.0
    ax_height_in = 10.0
    spacing_in = 0.5
    margin_left_in = 0.01
    margin_right_in = 0.01
    margin_bottom_in = 0.75
    margin_top_in = 0.01

    # --- 2. Calculate Total Figure Size ---
    total_width_in = (margin_left_in + ax_width_in * 3 + spacing_in * 2 + margin_right_in)
    total_height_in = margin_bottom_in + ax_height_in + margin_top_in

    # --- 3. Create the Figure ---
    fig = plt.figure(figsize=(total_width_in, total_height_in))

    # --- Define Plot-Specific Parameters ---
    p1 = [0.3, -0.7, -0.7, 0.3]
    v1_1 = np.array([1.0, 0.7, 0.7, 1.])
    v1_2 = np.array([-0.3, 0.6, -1.0, 0.7])
    b1 = 5.5

    p2 = np.array([0.0, 0.7, 0.0, 0.0])
    v2_1 = np.array([1.0, 1.7, 1.0, 1.0])
    v2_2 = np.array([-1.0, 1.0, 1.0, -1.0])
    b2 = 8

    p3 = np.array([0, 0, 0, 0])
    v3_1 = np.array([1, 1, 1, 1])
    v3_2 = np.array([1, -1, -1, 1])
    b3 = 10

    # --- NEW: Define an M matrix for the RG flow ---
    # This example matrix creates a saddle point, with flow moving away from the origin
    # along the last two dimensions and towards the origin along the first two.
    a = (64 * (-40 + np.pi ** 2)) / (3 * (20 + np.pi ** 2) ** 2)
    b = ((32 * (-20 + np.pi ** 2)) / (20 + np.pi ** 2) ** 2)
    M_flow = np.eye(4) * 2 + np.array([
        [a, -b, 0, 0],
        [-b, a, 0, 0],
        [0, 0, a, b],
        [0, 0, b, a]]
    )

    # --- Organize parameters into lists for iteration ---
    ps = [p1, p2, p3]
    v1s = [v1_1, v2_1, v3_1]
    v2s = [v1_2, v2_2, v3_2]
    bs = [b1, b2, b3]
    # Apply flow only to the second plot (index 1) for demonstration
    add_flows = [True, True, True]
    Ms = [M_flow, M_flow, M_flow]

    # --- 4. Create and Place Each Axes Manually ---
    for i in range(3):
        left_frac = (margin_left_in + i * (ax_width_in + spacing_in)) / total_width_in
        bottom_frac = margin_bottom_in / total_height_in
        width_frac = ax_width_in / total_width_in
        height_frac = ax_height_in / total_height_in

        ax = fig.add_axes([left_frac, bottom_frac, width_frac, height_frac])

        # Call get_fig with the appropriate parameters for this subplot
        get_fig(fig, ax, ps[i], v1s[i], v2s[i], add_arrows=False,
                u_range=(-bs[i], bs[i]), v_range=(-bs[i], bs[i]),
                # Pass the flow parameters for this specific plot
                add_flow=add_flows[i],
                M=Ms[i])

        label = f'({chr(97 + i)})'
        ax.text(0.5, -0.04, label, transform=ax.transAxes,
                ha='center', va='center', fontsize=35, color='black')

    # --- 5. Save the Figure ---
    plt.savefig('combined_figure_with_flow.pdf')
    print("Successfully saved the combined figures to 'combined_figure_with_flow.pdf'")


if __name__ == '__main__':
    main()
