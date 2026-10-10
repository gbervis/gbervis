
from pathlib import Path
from html import escape
import numpy as np
from PIL import Image

# Run from the gbervis repository:
# python scripts/generate_profile_banner_v3.py

# Define project directories relative to this script.
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'assets'
SHAPES = OUT / 'shapes'
OUT.mkdir(parents=True, exist_ok=True)

# Animation and canvas configuration.
N_POINTS = 900
SECONDS_PER_STATE = 5.0
HOLD_SECONDS = 3.5
POINT_RADIUS = 1.45
CANVAS_W, CANVAS_H = 1100, 475

# Mathematical and computational figures used in the animation.
FIGURES = [
    ('INTEGRAL', 'integral.png'),
    ('NORMAL DISTRIBUTION', 'bellshape.png'),
    ('ANSCOMBE QUARTET', 'anscombe.png'),
    ('MARKOV CHAIN', 'markov.png'),
    ('PARETO FRONT', 'pareto.png'),
    ('DATA INTEGRATION', 'etl.png'),
    ('NEURAL NETWORK', 'net.png'),
]

# Color palettes for GitHub's dark and light themes.
PALETTES = {
    'dark': dict(bg='#0B1220', bar='#111C2C', panel='#101C2D',
                 border='#29415D', text='#E4EDF8', muted='#96ABC2',
                 accent='#69B9FF', point='#70BEFF', cursor='#A3DDFF'),
    'light': dict(bg='#F7FAFE', bar='#E9F1FA', panel='#FFFFFF',
                  border='#BFD4E9', text='#1C3550', muted='#627F9C',
                  accent='#126AAE', point='#197BC2', cursor='#126AAE'),
}


def image_to_point_cloud(
    path,
    n_points=900,
    threshold=180,
    invert=False,
    padding=0.08,
    seed=42
):
    """
    Convert an image into a normalized point cloud.

    Dark pixels are selected by default. Setting invert=True
    selects bright pixels instead.

    The original aspect ratio is preserved, and the resulting
    coordinates are normalized to the [0, 1] interval.

    Parameters
    ----------
    path : str or Path
        Path to the source image.
    n_points : int
        Number of points to sample.
    threshold : int
        Grayscale threshold used to identify the figure.
    invert : bool
        Whether to select bright instead of dark pixels.
    padding : float
        Relative padding around the normalized figure.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 2) containing normalized
        coordinates.
    """

    rng = np.random.default_rng(seed)

    # Load the image and convert it to grayscale.
    img = Image.open(path).convert("L")
    arr = np.asarray(img)

    # Identify pixels belonging to the figure.
    mask = arr > threshold if invert else arr < threshold

    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise ValueError(f"No valid pixels detected in {path}")

    # Determine the figure's bounding box.
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x = xs.astype(float)
    y = ys.astype(float)

    # Use uniform scaling to preserve the original aspect ratio.
    scale = max(x_max - x_min, y_max - y_min, 1)

    # Center and normalize coordinates while preserving proportions.
    x = 0.5 + (x - (x_min + x_max) / 2) / scale * (1 - 2 * padding)
    y = 0.5 + (y - (y_min + y_max) / 2) / scale * (1 - 2 * padding)

    points = np.column_stack([x, y])

    # Randomly sample the requested number of points.
    # Sampling with replacement is used only when necessary.
    indices = rng.choice(
        len(points),
        size=n_points,
        replace=len(points) < n_points
    )

    points = points[indices]

    # Sort points spatially to support smoother transitions.
    order = np.lexsort((points[:, 1], points[:, 0]))

    return np.clip(points[order], 0, 1)


def load_states():
    """
    Load all source images and convert them into point clouds.

    Returns
    -------
    list[np.ndarray]
        List of point clouds, one for each animation state.
    """

    states = []
    missing = []

    for index, (name, filename) in enumerate(FIGURES):
        path = SHAPES / filename

        if not path.is_file():
            missing.append(str(path))
            continue

        states.append(
            image_to_point_cloud(
                path,
                n_points=N_POINTS,
                seed=42 + index
            )
        )

    if missing:
        raise FileNotFoundError(
            'Missing image files:\n' + '\n'.join(missing)
        )

    return states


def banner(theme, states):
    """
    Generate an animated SVG banner for the selected color theme.

    The left panel continuously morphs between mathematical
    point clouds, while the right terminal executes only once
    and preserves its final state.
    """

    colors = PALETTES[theme]
    count = len(states)
    duration = count * SECONDS_PER_STATE

    # Calculate normalized animation keyframes.
    key_times = []

    for i in range(count):
        key_times.extend((
            i * SECONDS_PER_STATE / duration,
            (i * SECONDS_PER_STATE + HOLD_SECONDS) / duration
        ))

    key_times.append(1.0)

    kt = ';'.join(f'{value:.7f}' for value in key_times)

    p = colors

    # Initialize the SVG document.
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" '
        f'width="{CANVAS_W}" height="{CANVAS_H}" '
        'role="img" '
        'aria-label="gbervis: animated mathematical point cloud and terminal profile">'
    ]

    # Define the terminal typography.
    s.append(
        '<style>'
        'text{font-family:Consolas,Menlo,DejaVu Sans Mono,monospace}'
        '</style>'
    )

    # Draw the main window and its title bar.
    s.append(
        f'<rect x="1" y="1" width="1098" height="473" '
        f'rx="17" fill="{p["bg"]}" '
        f'stroke="{p["border"]}" stroke-width="2"/>'
    )

    s.append(
        f'<path d="M18 1H1082Q1099 1 1099 18V42H1V18Q1 1 18 1Z" '
        f'fill="{p["bar"]}"/>'
    )

    # Draw the window control buttons.
    for x, c in [
        (22, '#FF6159'),
        (41, '#FFBD2E'),
        (60, '#28C840')
    ]:
        s.append(
            f'<circle cx="{x}" cy="22" r="5" fill="{c}"/>'
        )

    # Draw the window title.
    s.append(
        f'<text x="550" y="27" text-anchor="middle" '
        f'font-size="13" fill="{p["muted"]}">'
        'bash — gbervis@github'
        '</text>'
    )

    # Draw the left and right panels.
    s.append(
        f'<rect x="20" y="59" width="426" height="393" '
        f'rx="10" fill="{p["panel"]}" '
        f'stroke="{p["border"]}"/>'
    )

    s.append(
        f'<rect x="459" y="59" width="621" height="393" '
        f'rx="10" fill="{p["panel"]}" '
        f'stroke="{p["border"]}"/>'
    )

    # Left panel: title and animation metadata.
    s.append(
        f'<text x="36" y="86" font-size="17" '
        f'font-weight="bold" fill="{p["accent"]}">'
        'MODEL.SPACE'
        '</text>'
    )

    s.append(
        f'<text x="429" y="84" text-anchor="end" '
        f'font-size="11" fill="{p["muted"]}">'
        f'{N_POINTS} pts / {count} states'
        '</text>'
    )

    # Right panel: terminal title and GitHub username.
    s.append(
        f'<text x="476" y="86" font-size="17" '
        f'font-weight="bold" fill="{p["accent"]}">'
        'profile.sh'
        '</text>'
    )

    s.append(
        f'<text x="1062" y="84" text-anchor="end" '
        f'font-size="12" fill="{p["muted"]}">'
        '@gbervis'
        '</text>'
    )

    # ---------------------------------------------------------
    # LEFT PANEL: ANIMATED MATHEMATICAL POINT CLOUD
    # ---------------------------------------------------------

    # Apply a common transformation to the seven source figures.
    # Coordinates are converted from [0, 1] into a 360-unit space.
    s.append(
        '<g transform="translate(57 103) scale(0.95 0.90)">'
    )

    for j in range(N_POINTS):

        # Build each particle's trajectory through all states.
        route = []

        for state in states:
            route.extend((state[j], state[j]))

        # Return to the initial state to create a seamless loop.
        route.append(states[0][j])

        xs = ';'.join(
            f'{v[0] * 360:.2f}' for v in route
        )

        ys = ';'.join(
            f'{v[1] * 360:.2f}' for v in route
        )

        x0, y0 = states[0][j] * 360

        # Animate the X and Y coordinates independently.
        s.append(
            f'<circle cx="{x0:.2f}" cy="{y0:.2f}" '
            f'r="{POINT_RADIUS}" '
            f'fill="{p["point"]}" opacity="0.88">'

            f'<animate attributeName="cx" '
            f'values="{xs}" keyTimes="{kt}" '
            f'dur="{duration:.1f}s" '
            f'repeatCount="indefinite" calcMode="linear"/>'

            f'<animate attributeName="cy" '
            f'values="{ys}" keyTimes="{kt}" '
            f'dur="{duration:.1f}s" '
            f'repeatCount="indefinite" calcMode="linear"/>'

            '</circle>'
        )

    s.append('</g>')

    # ---------------------------------------------------------
    # LEFT PANEL: SYNCHRONIZED FIGURE LABELS
    # ---------------------------------------------------------

    # Display the label corresponding to the current figure.
    label_times = ';'.join(
        f'{j / count:.7f}' for j in range(count + 1)
    )

    for k, (name, _) in enumerate(FIGURES):

        opacity_values = ';'.join(
            ['1' if j == k else '0' for j in range(count)]
            + ['1' if k == 0 else '0']
        )

        s.append(
            f'<text x="36" y="436" font-size="12" '
            f'fill="{p["muted"]}" '
            f'opacity="{1 if k == 0 else 0}">'
            f'{escape(name)}'

            f'<animate attributeName="opacity" '
            f'values="{opacity_values}" '
            f'keyTimes="{label_times}" '
            f'calcMode="discrete" '
            f'dur="{duration:.1f}s" '
            f'repeatCount="indefinite"/>'

            '</text>'
        )

    # ---------------------------------------------------------
    # RIGHT PANEL: TERMINAL PROFILE
    # ---------------------------------------------------------

    # Terminal lines appear only once.
    # Their final visibility is preserved using fill="freeze".
    # Empty strings introduce intentional vertical spacing.
    rows = [
        ('gbervis@github:~$ whoami', 'cmd'),
        ('> Giusseppe Benito Bervis Quintero', 'text'),
        ('> Mathematician | Data & ML Builder', 'text'),
        ('', 'text'),

        ('$ cat interests', 'cmd'),
        ('  mathematical modelling · optimization · simulation', 'text'),
        ('  ML / AI · data science -> data engineering ', 'text'),
        ('', 'text'),

        ('$ echo $MISSION', 'cmd'),
        (' Exploring patterns and modelling complexity' + 
         ' to build meaningful solutions.', 'text'),
        ('', 'text'),

        ('$ echo $STATUS', 'cmd'),
        ('  always learning', 'text'),
    ]

    start_y, step_y = 121, 21

    # Reveal terminal lines progressively.
    for i, (line, kind) in enumerate(rows):

        if not line:
            continue

        color = p['accent'] if kind == 'cmd' else p['text']
        y = start_y + i * step_y

        s.append(
            f'<text x="482" y="{y}" font-size="14" '
            f'fill="{color}" opacity="0">'
            f'{escape(line)}'

            f'<set attributeName="opacity" to="1" '
            f'begin="{0.5 + i * 0.58:.2f}s" '
            f'fill="freeze"/>'

            '</text>'
        )

    # Calculate when the terminal execution finishes.
    end_at = 0.5 + (len(rows) - 1) * 0.58 + 0.6

    # Display the final terminal prompt.
    s.append(
        f'<text x="482" y="420" font-size="15" '
        f'fill="{p["accent"]}" opacity="0">'
        'gbervis@github:~$'

        f'<set attributeName="opacity" to="1" '
        f'begin="{end_at:.2f}s" fill="freeze"/>'

        '</text>'
    )

    # Keep the cursor blinking after the script has finished.
    s.append(
        f'<rect x="650" y="406" width="10" height="17" '
        f'fill="{p["cursor"]}" opacity="0">'

        f'<animate attributeName="opacity" '
        f'values="1;0;1" dur="1.2s" '
        f'begin="{end_at:.2f}s" '
        f'repeatCount="indefinite"/>'

        '</rect>'
    )

    # Close the SVG document.
    s.append('</svg>')

    # Save the generated banner to the assets directory.
    target = OUT / f'banner-{theme}.svg'

    target.write_text(
        '\n'.join(s),
        encoding='utf-8'
    )

    print(
        f"Generated: {target} "
        f"({target.stat().st_size / 1024:.1f} KB)"
    )


# -------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------

if __name__ == '__main__':

    print("Loading mathematical figures...")

    states = load_states()

    print(
        f"Successfully loaded {len(states)} figures "
        f"with {N_POINTS} points each."
    )

    for mode in ('light', 'dark'):
        print(f"Generating {mode} theme...")
        banner(mode, states)

    print("Banner generation completed successfully.")
    print("Open assets/banner-light.svg or assets/banner-dark.svg in your browser.")
