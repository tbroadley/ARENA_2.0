# %%

import os
import sys
import torch as t
from torch import Tensor
from einops import repeat, reduce
from ipywidgets import interact
import plotly.express as px
from pathlib import Path
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_ray_tracing"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"

# %%

def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    ys = t.linspace(-y_limit, y_limit, num_pixels)
    origins = t.zeros((num_pixels, 3))
    dests = t.stack([t.ones_like(ys), ys, t.zeros_like(ys)], dim=1)
    result = t.stack([origins, dests], dim=1)
    print(result)
    return result


rays1d = make_rays_1d(9, 10.0)

if MAIN:
    fig = render_lines_with_plotly(rays1d)


# %%

segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])

if MAIN:
    fig = render_lines_with_plotly(rays1d, segments)

# %%

@jaxtyped
def intersect_ray_1d(ray: Float[Tensor, "points=2 dim=3"], segment: Float[Tensor, "points=2 dim=3"]) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    D = ray[1, :2]
    A = t.stack((D, (segment[1] - segment[0])[:2])).T
    B = (segment[1] - ray[0])[:2]
    try:
        solution = t.linalg.solve(A, B)
    except t.linalg.LinAlgError:
        return False
    return solution[0] >= 0 and solution[1] >= 0 and solution[1] <= 1


if MAIN:
    tests.test_intersect_ray_1d(intersect_ray_1d)
    tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %%

def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    repeated_rays = repeat(rays, "nrays ... -> nrays nsegments ...", nsegments=segments.shape[0])
    repeated_segments = repeat(segments, "nsegments ... -> nrays nsegments ...", nrays=rays.shape[0])

    Ds = repeated_rays[..., 1, :2]

    L1s = repeated_segments[..., 0, :2]
    L2s = repeated_segments[..., 1, :2]

    As = t.stack((Ds, (L1s - L2s)), dim=2).mT
    
    dets = t.linalg.det(As)
    is_singular = dets.abs() < 1e-6
    As[is_singular] = t.eye(2)

    Os = repeated_rays[..., 0, :2]

    Bs = (L1s - Os)

    solutions = t.linalg.solve(As, Bs)

    us = solutions[..., 0]
    vs = solutions[..., 1]

    results = (us >= 0) & (vs >= 0) & (vs <= 1) & (~is_singular)
    result = results.any(dim=1)
    return result


if MAIN:
    # fig = render_lines_with_plotly(rays1d, segments)
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %%

def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[t.Tensor, "nrays 2 3"]:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    pass

    ys = repeat(t.linspace(-y_limit, y_limit, num_pixels_y), "y -> y z", z=num_pixels_z)
    zs = repeat(t.linspace(-z_limit, z_limit, num_pixels_z), "z -> y z", y=num_pixels_y)
    origins = t.zeros((num_pixels_y, num_pixels_z, 3))
    dests = t.stack([t.ones_like(ys), ys, zs], dim=2)
    result = t.stack([origins, dests], dim=2).flatten(0, 1)
    return result


if MAIN:
    rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
    render_lines_with_plotly(rays_2d)

# %%

if MAIN:
    one_triangle = t.tensor([[0, 0, 0], [3, 0.5, 0], [2, 3, 0]])
    A, B, C = one_triangle
    x, y, z = one_triangle.T

    fig = setup_widget_fig_triangle(x, y, z)

@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def response(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.data[2].update({"x": [P[0]], "y": [P[1]]})


if MAIN:
    display(fig)

# %%

Point = Float[Tensor, "points=3"]

@jaxtyped
# @typeguard.typechecked
def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    '''

    mat = t.stack((-D, B - A, C - A), dim=1)
    vec = O - A
    s, u, v = t.linalg.solve(mat, vec)
    result = (s >= 0 and u >= 0 and v >= 0 and u + v <= 1).item()
    print(result)
    return result


if MAIN:
    tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %%

def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    NR = rays.shape[0]
    assert rays.shape == (NR, 2, 3)

    triangles = repeat(triangle, "... -> nrays ...", nrays=NR)
    assert triangles.shape == (NR, 3, 3)

    A, B, C = triangles.unbind(dim=1)
    assert A.shape == B.shape == C.shape == (NR, 3)

    O, D = rays.unbind(dim=1)
    assert O.shape == D.shape == (NR, 3)

    mat = t.stack((-D, B - A, C - A), dim=2)
    assert mat.shape == (NR, 3, 3)

    vec = O - A
    assert vec.shape == (NR, 3)

    s, u, v = t.linalg.solve(mat, vec).unbind(dim=1)
    assert s.shape == u.shape == v.shape == (NR,)

    results = (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)
    assert results.shape == (NR,)
    return results


if MAIN:
    A = t.tensor([1, 0.0, -0.5])
    B = t.tensor([1, -0.5, 0.0])
    C = t.tensor([1, 0.5, 0.5])
    num_pixels_y = num_pixels_z = 25
    y_limit = z_limit = 0.5

    # Plot triangle & rays
    test_triangle = t.stack([A, B, C], dim=0)
    rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
    render_lines_with_plotly(rays2d, triangle_lines)

    # Calculate and display intersections
    intersects = raytrace_triangle(rays2d, test_triangle)
    img = intersects.reshape(num_pixels_y, num_pixels_z).int()
    imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%

def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.shape[0]

    A, B, C = repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(1)

    mat = t.stack([- D, B - A, C - A], dim=2)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=1)

    return ((s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & (~is_singular))


intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%

if MAIN:
    with open(section_dir / "pikachu.pt", "rb") as f:
        triangles = t.load(f)

# %%

def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    NR = rays.shape[0]
    NT = triangles.shape[0]

    repeated_rays = repeat(rays, "nr ... -> nr nt ...", nt=NT)
    assert repeated_rays.shape == (NR, NT, 2, 3)

    repeated_triangles = repeat(triangles, "nt ... -> nr nt ...", nr=NR)
    assert repeated_triangles.shape == (NR, NT, 3, 3)

    A, B, C = repeated_triangles.unbind(dim=2)
    assert A.shape == B.shape == C.shape == (NR, NT, 3)

    O, D = repeated_rays.unbind(dim=2)
    assert O.shape == D.shape == (NR, NT, 3)

    mat = t.stack((-D, B - A, C - A), dim=3)
    assert mat.shape == (NR, NT, 3, 3)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A
    assert vec.shape == (NR, NT, 3)

    s, u, v = t.linalg.solve(mat, vec).unbind(dim=2)
    assert s.shape == u.shape == v.shape == (NR, NT)

    in_triangle = (u >= 0) & (v >= 0) & (u + v <= 1) & (~is_singular)
    assert in_triangle.shape == (NR, NT)

    s[~in_triangle] = t.inf

    min_s = reduce(s, "nr nt -> nr", "min")
    assert min_s.shape == (NR,)

    return min_s


if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]): 
        fig.layout.annotations[i]['text'] = text
    fig.show()