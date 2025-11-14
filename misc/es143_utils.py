"""
es143_utils.py

Utility functions for ES143 camera calibration and AprilTag-based board detection.

Includes:
- Camera visualization in 3D using Plotly
- Drawing lines and conics in 2D with Matplotlib
- AprilTag detector setup and tag-to-board matching

Typical usage:
    from es143_utils import viz_board_calibration, detect_aprilboard

Requirements: numpy, plotly, cv2, scipy, pupil-apriltags
"""

import numpy as np

__all__ = [
    "viz_board_calibration",
    "add_plotly_camera",
    "add_mplot_line",
    "add_mplot_conic",
    "make_apriltag_detector",
    "detect_aprilboard",
    "AT_DETECTOR",
    "in2hom",
    "hom2in",
    "normalizing_transform"
]

# -------------------------------------------------------------------------
# plotly utilities
# -------------------------------------------------------------------------


def viz_board_calibration(
    calMatrix,
    calObjPoints,
    calRotations,
    calTranslations,
    img,
    raysize: float = 2.0,
    max_colors: int = 30,
):
    """
    Visualize a calibrated camera and its detected calibration boards in 3D using Plotly.

    Parameters
    ----------
    calMatrix : np.ndarray
        The 3×3 intrinsic camera matrix returned by cv2.calibrateCamera().
    calObjPoints : list of np.ndarray
        List of arrays (each Nx3) containing 3D object points for each calibration image.
    calRotations : list of np.ndarray
        List of rotation vectors (Rodrigues form) for each calibration image.
    calTranslations : list of np.ndarray
        List of translation vectors for each calibration image.
    img : np.ndarray
        One calibration image (only used to infer image dimensions for drawing the camera).
    raysize : float, optional
        Length of the 3D camera’s rays (default = 2.0).
    max_colors : int, optional
        Maximum number of unique colors to cycle through for calibration boards (default = 30).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        An interactive Plotly figure object showing:
          - the camera at the world origin (not shown in legend)
          - each calibration board as a colored plane
          - a legend for interactively toggling individual boards

    Notes
    -----
    - Each board is colored distinctly using a concatenated qualitative palette.
    - The legend is interactive: click a board name to show/hide it.
    - The first two traces (camera elements) are hidden from the legend automatically.
    """
    import cv2
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.spatial import ConvexHull

    # --- Initialize figure ---
    fig = go.Figure()
    fig.update_layout(
        scene=dict(
            xaxis_title="X (world)",
            yaxis_title="Y (world)",
            zaxis_title="Z (world)",
            aspectmode="data",
        ),
        width=900,
        height=700,
        legend=dict(
            title="Calibration Boards",
            itemsizing="constant",
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )

    camera = dict(
        eye=dict(x=-1.5, y=-0.1, z=-1.5),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=-1, z=0),
    )
    fig.update_layout(scene_camera=camera)

    # --- Draw camera at world origin ---
    P_world = calMatrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
    h, w = img.shape[:2]
    add_plotly_camera(h, w, P_world, raysize=raysize, figobj=fig)

    # Hide camera traces (assumes two were added: point + mesh)
    for idx in [0, 1]:
        if idx < len(fig.data):
            fig.data[idx].showlegend = False

    # --- Define distinct color palette ---
    palette = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.D3
        + px.colors.qualitative.Set3
        + px.colors.qualitative.Dark24
    )
    palette = palette[:max_colors]

    # --- Draw each calibration board ---
    for i, (obj_pts, rvec, tvec) in enumerate(
        zip(calObjPoints, calRotations, calTranslations)
    ):
        R, _ = cv2.Rodrigues(rvec)
        pts_world = (R @ obj_pts.T + tvec).T  # Nx3

        # Compute convex hull for filled polygon
        hull = ConvexHull(pts_world[:, :2])
        verts = pts_world[hull.vertices]

        color = palette[i % len(palette)]
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                opacity=0.6,
                color=color,
                name=f"Board {i}",
                hoverinfo="name",
                showlegend=True,
            )
        )

    return fig


def add_plotly_camera(h, w, camera, raysize, figobj):
    """
    Add a tetrahedral 3D camera model to a Plotly figure.

    Parameters
    ----------
    h : int
        Height of the image in pixels.
    w : int
        Width of the image in pixels.
    camera : np.ndarray
        3×4 camera projection matrix.
    raysize : float
        Length of tetrahedral edges (in world units).
    figobj : plotly.graph_objects.Figure
        Plotly Figure object to which the camera will be added.

    Returns
    -------
    plotly.graph_objects.Figure
        The same Figure object with the camera traces added.

    Notes
    -----
    - Follows the camera model described in Hartley & Zisserman,
      *Multiple View Geometry*, Section 6.2.3.
    - The function normalizes the camera so its principal ray has unit length
      and adds both a camera center marker and a tetrahedral frustum mesh.
    """
    import plotly.graph_objects as go

    # Normalize camera
    camera = (
        camera
        * np.sign(np.linalg.det(camera[:, 0:3]))
        / np.linalg.norm(camera[2, 0:3])
    )

    # Compute camera center (null space of P)
    _, _, v = np.linalg.svd(camera)
    C = np.transpose(v[-1, 0:3]) / v[-1, 3]

    # Back-project image corners
    S = np.array(
        [
            [0, 0, 1],
            [0, h - 1, 1],
            [w - 1, h - 1, 1],
            [w, 0, 1],
        ]
    )

    # Compute one 3D point along each ray
    X = np.transpose(
        np.linalg.lstsq(
            camera[:, 0:3],
            np.transpose(S) - np.expand_dims(camera[:, 3], axis=1),
            rcond=None,
        )[0]
    )

    # Unit vectors from camera center to each 3D point
    V = X - np.tile(C, (4, 1))
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    # Ensure vectors point forward
    V *= np.expand_dims(
        np.sign(np.sum(V * np.tile(camera[2, 0:3], (4, 1)), axis=1)), axis=1
    )

    # Scale to desired ray length
    V = np.tile(C, (4, 1)) + raysize * V

    # Append camera center
    V = np.vstack([C, V])

    # Add camera center marker
    figobj.add_trace(
        go.Scatter3d(
            x=[C[0]],
            y=[C[1]],
            z=[C[2]],
            mode="markers",
            marker=dict(size=3, color="#ff7f0e"),
            showlegend=False,
        )
    )

    # Add tetrahedral frustum mesh
    figobj.add_trace(
        go.Mesh3d(
            x=V[:, 0],
            y=V[:, 1],
            z=V[:, 2],
            i=[0, 0, 0, 0],
            j=[1, 2, 3, 4],
            k=[2, 3, 4, 1],
            opacity=0.5,
            color="#ff7f0e",
            showlegend=False,
        )
    )

    return figobj

def add_plotly_plane(plane,figobj):
    """
    Add 3D plane to pyplot figure.

    Parameters:
    ----------
        plane (tuple): Tuple containing plane parameters (a, b, c, d)
        figobj : plotly.graph_objects.Figure
                 Plotly Figure object to which the camera will be added.

    Returns
    -------
    plotly.graph_objects.Figure
        The same Figure object with the camera traces added.

    Note:
        This function only works for planes in a particular volume. It
        will need to be modified to work for other volumes.
    """
    import plotly.graph_objects as go
    
    # create regular grid of X,Y,Z values that span the volume that is relevant to
    # these particular planes
    #   (numpy.mgrid trick: j (complex) in step size forces inclusion of stop value)
    X, Y, Z = np.mgrid[-5:5:40j, -9:5:40j, 10:21:40j]

    # evaluate the linear functions on this volumetric grid that is zero at the plane
    implicitplane = plane[0]*X + plane[1]*Y + plane[2]*Z + plane[3]

    # display using plotly's isosurface trace
    figobj.add_trace(go.Isosurface(
          x=X.flatten(),
          y=Y.flatten(),
          z=Z.flatten(),
          value=implicitplane.flatten(),
          isomin=0,
          isomax=0,
          opacity=0.3,
          colorscale='blues',
          showscale=False, # remove colorbar
          ))

    return figobj

# -------------------------------------------------------------------------
# matplotlib utilities
# -------------------------------------------------------------------------

def add_mplot_line(line, ax=None, imsize=(100, 100), color='r', lw=2, n_samples=400, **kwargs):
    """
    Add a 2D line to a Matplotlib axis using the contour trick.

    Parameters
    ----------
    line : array_like, shape (3,)
        Line parameters (a, b, c) representing the algebraic form:
        a*x + b*y + c = 0.
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axes to draw on. If None, a new figure and axes
        are created automatically.
    imsize : int or tuple of (height, width), optional
        Size of the plotting region. 
        - If an int, assumes a square plot of that width/height.
        - If a tuple, should be (height, width), e.g. im.shape[:2].
    color : str or tuple, optional
        Line color. Default is 'r' (red).
    lw : float, optional
        Line width for contour. Default is 2.
    n_samples : int, optional
        Number of samples per axis used to compute the contour grid. Higher
        means smoother lines. Default is 400.
    **kwargs :
        Additional keyword arguments passed to `ax.contour()` (e.g., linestyle).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object.
    ax : matplotlib.axes.Axes
        The Matplotlib axes object.
    cs : matplotlib.contour.QuadContourSet
        The contour set for the plotted line (useful to customize style later).

    Notes
    -----
    This uses the zero-level contour of the implicit function h(x,y) = a*x + b*y + c.
    It’s robust, simple, and avoids handling special cases for vertical lines.

    Example
    -------
    >>> fig, ax, cs = add_mplot_line([1, -1, -10], imsize=(480, 640), color='g', lw=1.5)
    >>> ax.set_title('Example line on a 640x480 image')
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Parse imsize ---
    if np.isscalar(imsize):
        height = width = int(imsize)
    else:
        height, width = imsize[:2]

    # --- Validate line ---
    line = np.asarray(line).ravel()
    if line.size != 3:
        raise ValueError("line must be a 3-element vector [a, b, c].")

    # --- Create or configure axes ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.set_aspect(1)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
    else:
        fig = ax.figure

    # --- Generate a grid ---
    x, y = np.meshgrid(
        np.linspace(0, width, n_samples),
        np.linspace(0, height, n_samples)
    )
    h = line[0]*x + line[1]*y + line[2]

    # --- Draw the zero contour (the line) ---
    cs = ax.contour(x, y, h, levels=[0], colors=[color], linewidths=lw, **kwargs)

    return fig, ax, cs

import numpy as np
import matplotlib.pyplot as plt

def add_mplot_conic(conic, ax=None, imsize=(100, 100), color='b', lw=2, n_samples=400, **kwargs):
    """
    Add a 2D conic section (ellipse, parabola, hyperbola, etc.) to a Matplotlib axis
    using the contour trick.

    Parameters
    ----------
    conic : array_like, shape (6,)
        Conic parameters (a, b, c, d, e, f) representing the algebraic form:
            a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0.
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axes to draw on. If None, a new figure and axes
        are created automatically.
    imsize : int or tuple of (height, width), optional
        Size of the plotting region. 
        - If an int, assumes a square plot of that width/height.
        - If a tuple, should be (height, width), e.g. im.shape[:2].
    color : str or tuple, optional
        Conic color. Default is 'b' (blue).
    lw : float, optional
        Line width for contour. Default is 2.
    n_samples : int, optional
        Number of samples per axis used to compute the contour grid. Higher
        means smoother conic curves. Default is 400.
    **kwargs :
        Additional keyword arguments passed to `ax.contour()` (e.g., linestyle).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object.
    ax : matplotlib.axes.Axes
        The Matplotlib axes object.
    cs : matplotlib.contour.QuadContourSet
        The contour set for the plotted conic (useful to customize style later).

    Notes
    -----
    This uses the zero-level contour of the implicit function:
        h(x,y) = a*x^2 + b*x*y + c*y^2 + d*x + e*y + f.
    It's robust, general, and works for ellipses, circles, parabolas, and hyperbolas.

    Example
    -------
    >>> fig, ax, cs = add_mplot_conic([1, 0, 1, 0, 0, -40000],
    ...                               imsize=(480, 640), color='m', lw=1.5)
    >>> ax.set_title('Example conic: x² + y² - 40000 = 0 (circle of radius 200)')
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Parse imsize ---
    if np.isscalar(imsize):
        height = width = int(imsize)
    else:
        height, width = imsize[:2]

    # --- Validate conic coefficients ---
    conic = np.asarray(conic).ravel()
    if conic.size != 6:
        raise ValueError("conic must be a 6-element vector [a, b, c, d, e, f].")

    # --- Create or configure axes ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.set_aspect(1)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
    else:
        fig = ax.figure

    # --- Generate a grid ---
    x, y = np.meshgrid(
        np.linspace(0, width, n_samples),
        np.linspace(0, height, n_samples)
    )

    # --- Compute implicit function for the conic ---
    h = (conic[0]*x**2 +
         conic[1]*x*y +
         conic[2]*y**2 +
         conic[3]*x +
         conic[4]*y +
         conic[5])

    # --- Draw the zero contour (the conic) ---
    cs = ax.contour(x, y, h, levels=[0], colors=[color], linewidths=lw, **kwargs)

    return fig, ax, cs

# -------------------------------------------------------------------------
# Geometry Utilities
# -------------------------------------------------------------------------

def in2hom(X):
    """
    Convert from inhomogeneous to homogeneous coordinates

    Args:
      X  - NxD numpy array, tyically with D=2 (rows (x,y)) or D=3 (rows (x,y,z))

    Returns:
      Xh - Nx(D+1) numpy array with an appended column of ones
    """

    return np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)

# Convert from homogeneous to inhomogeneous coordinates
def hom2in(Xh):
    """
    Convert from homogeneous to inhomogeneous coordinates

    Args:
      Xh  - Nx(D+1) numpy array, tyically with D=2 (rows (x,y,w)) or D=3 (rows (x,y,z,w))

    Returns:
      X - NxD numpy array with the first D columns divided by column (D+1)
    """

    return Xh[:, :2] / Xh[:, 2:]

def normalizing_transform(X):
    """
    Compute a (Dx1)x(Dx1) normalizing transformation from N points in D dimensions

    Args:
      X - NxD numpy array, tyically with D=2 (rows (x,y)) or D=3 (rows (x,y,z))

    Returns:
      T - a (D+1)x(D+1) matrix that normalizes the points to be centered at their
          centroid, with average distance equal to sqrt(D)
    """

    # dimension of datapoints
    D = X.shape[1]

    # Compute centroid
    centroid = X.mean(axis=0, keepdims=True)

    # Compute the denominator of scale factor s
    denom = np.mean(np.sqrt(np.sum((X - centroid) ** 2, axis=1)))

    # Compute s, tx, ty
    s = np.sqrt(D) / denom
    t = -s * centroid[0, :]
    t = np.concatenate((t,np.array([1])))

    # return matrix T
    return np.concatenate((s*np.eye(D+1,D+1)[:,0:D], np.expand_dims(t,1)),axis=1)


# -------------------------------------------------------------------------
# AprilTag detection utilities
# -------------------------------------------------------------------------


def make_apriltag_detector(**kwargs):
    """
    Create and configure an AprilTag detector instance.

    This factory function constructs a `pupil_apriltags.Detector` object
    using either default parameters or user-specified overrides. It is used
    internally to create the default global detector (`AT_DETECTOR`), but can
    also be called directly to create custom detectors with different
    performance tradeoffs.

    Parameters
    ----------
    **kwargs : keyword arguments, optional
        Optional overrides for the underlying `Detector()` constructor.
        Common parameters include:
            families : str
                Tag family name (default = 'tag36h11').
            nthreads : int
                Number of threads for parallel detection (default = 1).
            quad_decimate : float
                Image decimation factor; larger values (e.g., 2.0) improve
                speed at the cost of accuracy (default = 1.0).
            quad_sigma : float
                Gaussian blur sigma for detection preprocessing (default = 0.0).
            refine_edges : int
                Whether to refine tag edges (default = 1).
            decode_sharpening : float
                Amount of image sharpening during decoding (default = 0.25).
            debug : int
                Enable debug visualization (default = 0).

    Returns
    -------
    Detector
        A configured `pupil_apriltags.Detector` instance.

    Examples
    --------
    **Default usage (default detection parameters):**
        >>> from es143_utils import detect_aprilboard
        >>> imgpoints, objpoints, ids = detect_aprilboard(img_gray, at_board)

        This is suitable for most cases.

    **Custom usage (with user-configured detection parameters):**
        >>> from es143_utils import make_apriltag_detector, detect_aprilboard
        >>> fast_detector = make_apriltag_detector(quad_decimate=2.0, nthreads=4)
        >>> imgpoints, objpoints, ids = detect_aprilboard(img_gray, at_board, fast_detector)

        Here, a faster detector is created with image downsampling and multithreading
        for large or high-frame-rate inputs.

    Notes
    -----
    - The default detector (`AT_DETECTOR`) is created at import time for
      convenience and uses the parameters shown above.
    - To inspect available parameters, see the official AprilTag documentation.
    """
    try:
        from pupil_apriltags import Detector
    except ImportError:
        Detector = None
        print(
            "Warning: pupil_apriltags not installed; AprilTag functions will be unavailable."
        )

    if Detector is None:
        raise ImportError(
            "pupil_apriltags not installed. Install it with `pip install pupil_apriltags`."
        )

    defaults = dict(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )
    defaults.update(kwargs)
    return Detector(**defaults)


# Create a default detector instance at import time for quick use.
try:
    AT_DETECTOR = make_apriltag_detector()
except ImportError:
    AT_DETECTOR = None


def detect_aprilboard(img, board, apriltag_detector=None):
    """
    Detect AprilTags corresponding to a known calibration board in an image.

    This function detects AprilTags in a grayscale image using a
    `pupil_apriltags.Detector` instance and matches the detected tags to
    a known calibration board definition. It returns the corresponding
    image-space and object-space coordinates of detected tag centers, which
    are typically used for camera calibration or pose estimation.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image containing AprilTags. If a color image is provided,
        it will be converted to grayscale before detection.
    board : list of dict
        A list describing the AprilTag board layout. Each element must include:
            {'tag_id': int, 'center': (X, Y, Z)}
        where (X, Y, Z) are the 3D coordinates of the tag center in board
        reference units (e.g., inches or millimeters).
    apriltag_detector : pupil_apriltags.Detector, optional
        A configured AprilTag detector instance. Only needed for
        customization of the detector parameters via `make_apriltag_detector()`.

    Returns
    -------
    imgpoints : (N, 2) np.ndarray
        Image coordinates (x, y) of the detected tag centers.
    objpoints : (N, 3) np.ndarray
        Corresponding 3D object coordinates (X, Y, Z) of those tags on the board.
    tagIDs : list of int
        List of tag IDs successfully detected and matched between image and board.

    Examples
    --------
    **Default usage (default detection parameters):**
        >>> from es143_utils import detect_aprilboard
        >>> imgpoints, objpoints, ids = detect_aprilboard(img_gray, at_board)

        This is suitable for most cases.

    **Custom usage (with user-configured detection parameters):**
        >>> from es143_utils import make_apriltag_detector, detect_aprilboard
        >>> fast_detector = make_apriltag_detector(quad_decimate=2.0, nthreads=4)
        >>> imgpoints, objpoints, ids = detect_aprilboard(img_gray, at_board, fast_detector)

        Here, a faster detector is created with image downsampling and multithreading
        for large or high-frame-rate inputs.

    Notes
    -----
    - The function filters out detected tags that do not appear in the provided
      `board` definition, ensuring that only valid calibration targets are returned.
    - If no valid tags are found, all returned arrays will be empty.
    - To inspect detector parameters, see the official AprilTag documentation.
    """
    import cv2

    if apriltag_detector is None:
        apriltag_detector = AT_DETECTOR

    imgpoints, objpoints, tagIDs = [], [], []

    # Convert image to grayscale if necessary
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    imgtags = apriltag_detector.detect(
        img, estimate_tag_pose=False, camera_params=None, tag_size=None
    )

    if len(imgtags):
        # Board and image tag IDs
        brdtagIDs = [sub["tag_id"] for sub in board]
        imgtagIDs = [sub.tag_id for sub in imgtags]

        # Intersection
        tagIDs = list(set(brdtagIDs).intersection(imgtagIDs))

        if len(tagIDs):
            # Matching board entries
            objs = [sub for sub in board if sub["tag_id"] in tagIDs]
            objpoints = np.vstack([sub["center"] for sub in objs])

            # Matching image detections
            imgs = [sub for sub in imgtags if sub.tag_id in tagIDs]
            imgpoints = np.vstack([sub.center for sub in imgs])

    return imgpoints, objpoints, tagIDs
