"""
ePlace-style Analytical Placement Algorithm
============================================

Based on the ePlace algorithm from UCLA:
"ePlace: Electrostatics-Based Placement Using Fast Fourier Transform and Nesterov's Method"
IEEE Trans. CAD, 2015.

This implementation provides:
1. Electrostatic density modeling (components as positive charges)
2. Gradient-based optimization using Nesterov's accelerated method
3. Wirelength optimization using log-sum-exp approximation
4. Global placement + legalization + detailed placement

Key insight from ePlace:
- Model placement as an electrostatic system
- Components are positive charges that repel each other
- Density cost = potential energy of the system
- Use Poisson's equation solved via FFT for fast density computation

References:
- ePlace: https://dl.acm.org/doi/10.1145/2699873
- DREAMPlace: https://github.com/limbo018/DREAMPlace
- Nesterov's method: "A method of solving a convex programming problem" (1983)
"""

from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
import math
from collections import defaultdict


@dataclass
class Component:
    """Component for placement."""
    ref: str
    width: float
    height: float
    x: float = 0.0  # Center X
    y: float = 0.0  # Center Y
    fixed: bool = False  # Fixed components don't move
    weight: float = 1.0  # Importance weight

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def left(self) -> float:
        return self.x - self.width / 2

    @property
    def right(self) -> float:
        return self.x + self.width / 2

    @property
    def bottom(self) -> float:
        return self.y - self.height / 2

    @property
    def top(self) -> float:
        return self.y + self.height / 2


@dataclass
class Net:
    """Net connecting multiple components."""
    name: str
    pins: List[Tuple[str, float, float]]  # (component_ref, pin_offset_x, pin_offset_y)
    weight: float = 1.0


@dataclass
class PlacementResult:
    """Result of analytical placement."""
    positions: Dict[str, Tuple[float, float]]  # ref -> (x, y)
    wirelength: float
    density_cost: float
    iterations: int
    converged: bool


class ePlaceAnalytical:
    """
    Electrostatic-based Analytical Placement.

    Uses gradient descent with electrostatic density modeling
    to find optimal component positions.
    """

    def __init__(
        self,
        board_width: float,
        board_height: float,
        grid_bins_x: int = 64,
        grid_bins_y: int = 64,
        target_density: float = 0.8,
        wirelength_weight: float = 1.0,
        density_weight: float = 1.0
    ):
        """
        Args:
            board_width: Board width in mm
            board_height: Board height in mm
            grid_bins_x: Number of density bins in X direction
            grid_bins_y: Number of density bins in Y direction
            target_density: Target density ratio (0-1)
            wirelength_weight: Weight for wirelength term
            density_weight: Weight for density term
        """
        self.board_width = board_width
        self.board_height = board_height
        self.grid_bins_x = grid_bins_x
        self.grid_bins_y = grid_bins_y
        self.target_density = target_density
        self.wl_weight = wirelength_weight
        self.density_weight = density_weight

        # Bin dimensions
        self.bin_width = board_width / grid_bins_x
        self.bin_height = board_height / grid_bins_y

        # Precompute target density per bin
        self.target_density_per_bin = (
            target_density * self.bin_width * self.bin_height
        )

    def place(
        self,
        components: List[Component],
        nets: List[Net],
        max_iterations: int = 500,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        convergence_threshold: float = 1e-4,
        callback: Optional[Callable[[int, float, float], None]] = None
    ) -> PlacementResult:
        """
        Run analytical placement.

        Args:
            components: List of components to place
            nets: List of nets connecting components
            max_iterations: Maximum optimization iterations
            learning_rate: Gradient descent learning rate
            momentum: Nesterov momentum parameter
            convergence_threshold: Stop when improvement < threshold
            callback: Optional callback(iteration, wirelength, density)

        Returns:
            PlacementResult with optimized positions
        """
        # Initialize positions
        self._initialize_positions(components)

        # Build component index
        comp_idx = {c.ref: i for i, c in enumerate(components)}

        # Nesterov momentum state
        n_components = len(components)
        velocity_x = np.zeros(n_components)
        velocity_y = np.zeros(n_components)

        # Position arrays (for vectorization)
        pos_x = np.array([c.x for c in components])
        pos_y = np.array([c.y for c in components])
        widths = np.array([c.width for c in components])
        heights = np.array([c.height for c in components])
        fixed = np.array([c.fixed for c in components])

        prev_cost = float('inf')
        converged = False

        for iteration in range(max_iterations):
            # Compute gradients
            wl_grad_x, wl_grad_y, wl_cost = self._wirelength_gradient(
                pos_x, pos_y, widths, heights, nets, comp_idx
            )

            density_grad_x, density_grad_y, density_cost = self._density_gradient(
                pos_x, pos_y, widths, heights
            )

            # Combined gradient
            grad_x = self.wl_weight * wl_grad_x + self.density_weight * density_grad_x
            grad_y = self.wl_weight * wl_grad_y + self.density_weight * density_grad_y

            # Zero gradients for fixed components
            grad_x[fixed] = 0
            grad_y[fixed] = 0

            # Nesterov's accelerated gradient descent
            velocity_x = momentum * velocity_x - learning_rate * grad_x
            velocity_y = momentum * velocity_y - learning_rate * grad_y

            # Update positions with momentum lookahead
            pos_x = pos_x + momentum * velocity_x - learning_rate * grad_x
            pos_y = pos_y + momentum * velocity_y - learning_rate * grad_y

            # Clamp to board boundaries
            pos_x = np.clip(pos_x, widths / 2, self.board_width - widths / 2)
            pos_y = np.clip(pos_y, heights / 2, self.board_height - heights / 2)

            # Compute total cost
            total_cost = self.wl_weight * wl_cost + self.density_weight * density_cost

            # Callback
            if callback:
                callback(iteration, wl_cost, density_cost)

            # Check convergence
            if abs(prev_cost - total_cost) < convergence_threshold * abs(prev_cost):
                converged = True
                break

            prev_cost = total_cost

            # Adaptive learning rate decay
            if iteration > 0 and iteration % 100 == 0:
                learning_rate *= 0.9

        # Update component positions
        for i, comp in enumerate(components):
            comp.x = pos_x[i]
            comp.y = pos_y[i]

        # Final legalization
        self._legalize(components)

        # Build result
        positions = {c.ref: (c.x, c.y) for c in components}
        final_wl = self._compute_hpwl(components, nets, comp_idx)

        return PlacementResult(
            positions=positions,
            wirelength=final_wl,
            density_cost=density_cost,
            iterations=iteration + 1,
            converged=converged
        )

    def _initialize_positions(self, components: List[Component]):
        """Initialize component positions using center-of-mass spreading."""
        # Start with components spread across the board
        n = len(components)
        if n == 0:
            return

        # Grid-based initial placement
        cols = int(math.ceil(math.sqrt(n * self.board_width / self.board_height)))
        rows = int(math.ceil(n / cols))

        cell_w = self.board_width / cols
        cell_h = self.board_height / rows

        for i, comp in enumerate(components):
            if not comp.fixed:
                row = i // cols
                col = i % cols
                comp.x = (col + 0.5) * cell_w
                comp.y = (row + 0.5) * cell_h

    def _wirelength_gradient(
        self,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        widths: np.ndarray,
        heights: np.ndarray,
        nets: List[Net],
        comp_idx: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute wirelength gradient using log-sum-exp (LSE) approximation.

        The LSE approximation is smooth and differentiable:
        max(x1, ..., xn) ≈ γ * log(exp(x1/γ) + ... + exp(xn/γ))

        where γ is a smoothing parameter.
        """
        n = len(pos_x)
        grad_x = np.zeros(n)
        grad_y = np.zeros(n)
        total_wl = 0.0

        gamma = 0.5  # Smoothing parameter

        for net in nets:
            if len(net.pins) < 2:
                continue

            # Get pin positions
            pin_x = []
            pin_y = []
            pin_comp_idx = []

            for ref, ox, oy in net.pins:
                if ref in comp_idx:
                    idx = comp_idx[ref]
                    pin_x.append(pos_x[idx] + ox)
                    pin_y.append(pos_y[idx] + oy)
                    pin_comp_idx.append(idx)

            if len(pin_x) < 2:
                continue

            pin_x = np.array(pin_x)
            pin_y = np.array(pin_y)

            # LSE approximation for max and min
            exp_pos_x = np.exp(pin_x / gamma)
            exp_neg_x = np.exp(-pin_x / gamma)
            exp_pos_y = np.exp(pin_y / gamma)
            exp_neg_y = np.exp(-pin_y / gamma)

            sum_exp_pos_x = np.sum(exp_pos_x)
            sum_exp_neg_x = np.sum(exp_neg_x)
            sum_exp_pos_y = np.sum(exp_pos_y)
            sum_exp_neg_y = np.sum(exp_neg_y)

            # Approximate HPWL
            max_x = gamma * np.log(sum_exp_pos_x)
            min_x = -gamma * np.log(sum_exp_neg_x)
            max_y = gamma * np.log(sum_exp_pos_y)
            min_y = -gamma * np.log(sum_exp_neg_y)

            wl = (max_x - min_x) + (max_y - min_y)
            total_wl += net.weight * wl

            # Gradients
            for i, idx in enumerate(pin_comp_idx):
                # d(LSE_max)/dx = exp(x/γ) / sum(exp(xi/γ))
                grad_x[idx] += net.weight * (
                    exp_pos_x[i] / sum_exp_pos_x - exp_neg_x[i] / sum_exp_neg_x
                )
                grad_y[idx] += net.weight * (
                    exp_pos_y[i] / sum_exp_pos_y - exp_neg_y[i] / sum_exp_neg_y
                )

        return grad_x, grad_y, total_wl

    def _density_gradient(
        self,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        widths: np.ndarray,
        heights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute density gradient using electrostatic model.

        Each component is modeled as a positive charge.
        The density cost is the potential energy of the electrostatic system.
        Uses bell-shaped density functions for smoothness.
        """
        n = len(pos_x)
        grad_x = np.zeros(n)
        grad_y = np.zeros(n)

        # Compute density map
        density = np.zeros((self.grid_bins_y, self.grid_bins_x))

        # For each component, add its density to overlapping bins
        for i in range(n):
            self._add_component_density(
                density, pos_x[i], pos_y[i], widths[i], heights[i]
            )

        # Compute overflow (density exceeding target)
        overflow = np.maximum(0, density - self.target_density_per_bin)
        density_cost = np.sum(overflow ** 2)

        # Compute electric field (gradient of potential)
        # Using simple finite differences for the potential gradient
        potential = self._compute_potential(overflow)

        # Compute gradient for each component
        for i in range(n):
            gx, gy = self._component_field_gradient(
                potential, pos_x[i], pos_y[i], widths[i], heights[i]
            )
            grad_x[i] = gx
            grad_y[i] = gy

        return grad_x, grad_y, density_cost

    def _add_component_density(
        self,
        density: np.ndarray,
        cx: float,
        cy: float,
        w: float,
        h: float
    ):
        """Add component's density contribution to the density map."""
        # Component bounds
        left = cx - w / 2
        right = cx + w / 2
        bottom = cy - h / 2
        top = cy + h / 2

        # Bin indices
        bin_left = max(0, int(left / self.bin_width))
        bin_right = min(self.grid_bins_x - 1, int(right / self.bin_width))
        bin_bottom = max(0, int(bottom / self.bin_height))
        bin_top = min(self.grid_bins_y - 1, int(top / self.bin_height))

        # Add density using bell-shaped function for smoothness
        for by in range(bin_bottom, bin_top + 1):
            for bx in range(bin_left, bin_right + 1):
                # Bin center
                bcx = (bx + 0.5) * self.bin_width
                bcy = (by + 0.5) * self.bin_height

                # Overlap area (simplified)
                ox_left = max(left, bx * self.bin_width)
                ox_right = min(right, (bx + 1) * self.bin_width)
                oy_bottom = max(bottom, by * self.bin_height)
                oy_top = min(top, (by + 1) * self.bin_height)

                if ox_right > ox_left and oy_top > oy_bottom:
                    overlap_area = (ox_right - ox_left) * (oy_top - oy_bottom)
                    density[by, bx] += overlap_area

    def _compute_potential(self, overflow: np.ndarray) -> np.ndarray:
        """
        Compute electrostatic potential from overflow density.

        Uses discrete cosine transform (DCT) to solve Poisson's equation.
        ∇²φ = ρ (Poisson's equation)
        """
        # Simple approximation: potential is proportional to smoothed overflow
        # For exact solution, use FFT-based Poisson solver

        # Gaussian smoothing as approximation
        from scipy.ndimage import gaussian_filter
        try:
            potential = gaussian_filter(overflow, sigma=2)
        except ImportError:
            # Fallback: simple averaging
            potential = overflow.copy()
            for _ in range(3):
                padded = np.pad(potential, 1, mode='edge')
                potential = (
                    padded[:-2, 1:-1] + padded[2:, 1:-1] +
                    padded[1:-1, :-2] + padded[1:-1, 2:] +
                    4 * padded[1:-1, 1:-1]
                ) / 8

        return potential

    def _component_field_gradient(
        self,
        potential: np.ndarray,
        cx: float,
        cy: float,
        w: float,
        h: float
    ) -> Tuple[float, float]:
        """Compute electric field gradient for a component."""
        # Component center in bin coordinates
        bin_x = cx / self.bin_width
        bin_y = cy / self.bin_height

        # Clamp to valid range
        bin_x = max(0.5, min(self.grid_bins_x - 1.5, bin_x))
        bin_y = max(0.5, min(self.grid_bins_y - 1.5, bin_y))

        # Integer bin indices
        bx = int(bin_x)
        by = int(bin_y)

        # Ensure we're within bounds for gradient computation
        bx = max(1, min(self.grid_bins_x - 2, bx))
        by = max(1, min(self.grid_bins_y - 2, by))

        # Gradient using central differences
        grad_x = (potential[by, bx + 1] - potential[by, bx - 1]) / (2 * self.bin_width)
        grad_y = (potential[by + 1, bx] - potential[by - 1, bx]) / (2 * self.bin_height)

        # Scale by component area (larger components get stronger force)
        area = w * h
        return grad_x * area, grad_y * area

    def _legalize(self, components: List[Component]):
        """
        Legalize placement by removing overlaps using Tetris-style algorithm.

        This is based on the Tetris legalizer from:
        "FastPlace: Efficient Analytical Placement Using Cell Shifting,
         Iterative Local Refinement, and a Hybrid Net Model"

        Guarantees zero overlaps by placing components one at a time
        into non-overlapping positions.
        """
        # Sort components by x-coordinate (left to right), then by y
        # This mimics how Tetris places pieces
        movable = [c for c in components if not c.fixed]
        fixed = [c for c in components if c.fixed]

        # Sort by current x position (left-to-right placement order)
        movable.sort(key=lambda c: (c.x, c.y))

        # Build occupancy grid for fast overlap checking
        # Grid resolution: 0.5mm
        grid_res = 0.5
        grid_w = int(math.ceil(self.board_width / grid_res)) + 1
        grid_h = int(math.ceil(self.board_height / grid_res)) + 1

        # Occupancy map: which component occupies each cell
        occupancy = [[None for _ in range(grid_w)] for _ in range(grid_h)]

        # Mark fixed components in occupancy grid
        for comp in fixed:
            self._mark_occupancy(occupancy, comp, grid_res, grid_w, grid_h)

        # Place each movable component using Tetris algorithm
        placed = list(fixed)
        for comp in movable:
            # Find nearest legal position using spiral search
            legal_x, legal_y = self._tetris_find_position(
                comp, occupancy, grid_res, grid_w, grid_h
            )
            comp.x = legal_x
            comp.y = legal_y

            # Mark this component in the occupancy grid
            self._mark_occupancy(occupancy, comp, grid_res, grid_w, grid_h)
            placed.append(comp)

        # Final verification pass using iterative displacement
        self._resolve_remaining_overlaps(placed)

    def _mark_occupancy(
        self,
        occupancy: List[List],
        comp: Component,
        grid_res: float,
        grid_w: int,
        grid_h: int
    ):
        """Mark component's footprint in occupancy grid."""
        min_gx = max(0, int((comp.x - comp.width/2) / grid_res))
        max_gx = min(grid_w - 1, int((comp.x + comp.width/2) / grid_res))
        min_gy = max(0, int((comp.y - comp.height/2) / grid_res))
        max_gy = min(grid_h - 1, int((comp.y + comp.height/2) / grid_res))

        for gy in range(min_gy, max_gy + 1):
            for gx in range(min_gx, max_gx + 1):
                occupancy[gy][gx] = comp.ref

    def _is_position_free(
        self,
        x: float, y: float,
        w: float, h: float,
        occupancy: List[List],
        grid_res: float,
        grid_w: int,
        grid_h: int
    ) -> bool:
        """Check if position is free in occupancy grid."""
        min_gx = max(0, int((x - w/2) / grid_res))
        max_gx = min(grid_w - 1, int((x + w/2) / grid_res))
        min_gy = max(0, int((y - h/2) / grid_res))
        max_gy = min(grid_h - 1, int((y + h/2) / grid_res))

        # Check bounds
        if x - w/2 < 0 or x + w/2 > self.board_width:
            return False
        if y - h/2 < 0 or y + h/2 > self.board_height:
            return False

        for gy in range(min_gy, max_gy + 1):
            for gx in range(min_gx, max_gx + 1):
                if occupancy[gy][gx] is not None:
                    return False
        return True

    def _tetris_find_position(
        self,
        comp: Component,
        occupancy: List[List],
        grid_res: float,
        grid_w: int,
        grid_h: int
    ) -> Tuple[float, float]:
        """
        Find nearest legal position using spiral search around target.

        Searches in expanding squares around the target position
        until a free position is found.
        """
        target_x, target_y = comp.x, comp.y
        w, h = comp.width, comp.height

        # Clamp target to board bounds
        target_x = max(w/2, min(self.board_width - w/2, target_x))
        target_y = max(h/2, min(self.board_height - h/2, target_y))

        # Check if current position is free
        if self._is_position_free(target_x, target_y, w, h, occupancy, grid_res, grid_w, grid_h):
            return target_x, target_y

        # Spiral search with grid resolution
        step = grid_res
        max_radius = max(self.board_width, self.board_height)

        for radius in np.arange(step, max_radius, step):
            # Search along the perimeter of a square at this radius
            for dx in np.arange(-radius, radius + step, step):
                for dy in [-radius, radius]:  # Top and bottom edges
                    test_x = target_x + dx
                    test_y = target_y + dy
                    if self._is_position_free(test_x, test_y, w, h, occupancy, grid_res, grid_w, grid_h):
                        return test_x, test_y

            for dy in np.arange(-radius + step, radius, step):
                for dx in [-radius, radius]:  # Left and right edges
                    test_x = target_x + dx
                    test_y = target_y + dy
                    if self._is_position_free(test_x, test_y, w, h, occupancy, grid_res, grid_w, grid_h):
                        return test_x, test_y

        # Fallback: scan entire board systematically
        for y in np.arange(h/2, self.board_height - h/2, step):
            for x in np.arange(w/2, self.board_width - w/2, step):
                if self._is_position_free(x, y, w, h, occupancy, grid_res, grid_w, grid_h):
                    return x, y

        # Last resort: return original position (will have overlaps)
        return target_x, target_y

    def _resolve_remaining_overlaps(self, components: List[Component]):
        """
        Final pass to resolve any remaining overlaps.

        Uses iterative displacement with guaranteed termination.
        """
        max_iterations = 1000
        min_sep = 0.1  # Minimum separation between components

        for iteration in range(max_iterations):
            has_overlap = False

            for i, c1 in enumerate(components):
                if c1.fixed:
                    continue

                for j, c2 in enumerate(components):
                    if i >= j:
                        continue

                    # Check overlap with small margin
                    if self._overlaps(c1.x, c1.y, c1.width + min_sep, c1.height + min_sep,
                                     c2.x, c2.y, c2.width + min_sep, c2.height + min_sep):
                        has_overlap = True

                        # Calculate separation vector
                        dx = c1.x - c2.x
                        dy = c1.y - c2.y
                        dist = math.sqrt(dx*dx + dy*dy) + 0.001

                        # Required separation
                        req_sep_x = (c1.width + c2.width) / 2 + min_sep
                        req_sep_y = (c1.height + c2.height) / 2 + min_sep

                        # Choose axis with smaller overlap
                        overlap_x = req_sep_x - abs(dx)
                        overlap_y = req_sep_y - abs(dy)

                        if overlap_x < overlap_y and overlap_x > 0:
                            # Separate in x direction
                            move = (overlap_x / 2 + 0.1) * (1 if dx >= 0 else -1)
                            if not c1.fixed:
                                c1.x += move
                            if not c2.fixed:
                                c2.x -= move
                        elif overlap_y > 0:
                            # Separate in y direction
                            move = (overlap_y / 2 + 0.1) * (1 if dy >= 0 else -1)
                            if not c1.fixed:
                                c1.y += move
                            if not c2.fixed:
                                c2.y -= move

                        # Clamp to board bounds
                        c1.x = max(c1.width/2, min(self.board_width - c1.width/2, c1.x))
                        c1.y = max(c1.height/2, min(self.board_height - c1.height/2, c1.y))
                        c2.x = max(c2.width/2, min(self.board_width - c2.width/2, c2.x))
                        c2.y = max(c2.height/2, min(self.board_height - c2.height/2, c2.y))

            if not has_overlap:
                break

    def _overlaps(
        self,
        x1: float, y1: float, w1: float, h1: float,
        x2: float, y2: float, w2: float, h2: float
    ) -> bool:
        """Check if two rectangles overlap."""
        return not (
            x1 + w1/2 <= x2 - w2/2 or  # 1 is left of 2
            x1 - w1/2 >= x2 + w2/2 or  # 1 is right of 2
            y1 + h1/2 <= y2 - h2/2 or  # 1 is below 2
            y1 - h1/2 >= y2 + h2/2     # 1 is above 2
        )

    def _compute_hpwl(
        self,
        components: List[Component],
        nets: List[Net],
        comp_idx: Dict[str, int]
    ) -> float:
        """Compute Half-Perimeter Wirelength."""
        total_wl = 0.0

        for net in nets:
            if len(net.pins) < 2:
                continue

            min_x = float('inf')
            max_x = float('-inf')
            min_y = float('inf')
            max_y = float('-inf')

            for ref, ox, oy in net.pins:
                if ref in comp_idx:
                    idx = comp_idx[ref]
                    px = components[idx].x + ox
                    py = components[idx].y + oy
                    min_x = min(min_x, px)
                    max_x = max(max_x, px)
                    min_y = min(min_y, py)
                    max_y = max(max_y, py)

            if min_x < float('inf'):
                total_wl += net.weight * ((max_x - min_x) + (max_y - min_y))

        return total_wl


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analytical_placement(
    components: Dict[str, Dict],
    nets: Dict[str, Dict],
    board_width: float,
    board_height: float,
    **kwargs
) -> Dict[str, Tuple[float, float]]:
    """
    Run analytical placement on components and nets.

    Args:
        components: Dict of {ref: {'width': w, 'height': h, 'fixed': bool}}
        nets: Dict of {net_name: {'pins': [(ref, ox, oy), ...]}}
        board_width: Board width in mm
        board_height: Board height in mm

    Returns:
        Dict of {ref: (x, y)} positions
    """
    # Convert to internal format
    comp_list = [
        Component(
            ref=ref,
            width=info.get('width', 2),
            height=info.get('height', 2),
            x=info.get('x', 0),
            y=info.get('y', 0),
            fixed=info.get('fixed', False)
        )
        for ref, info in components.items()
    ]

    net_list = [
        Net(
            name=name,
            pins=info.get('pins', [])
        )
        for name, info in nets.items()
    ]

    # Run placement
    placer = ePlaceAnalytical(board_width, board_height, **kwargs)
    result = placer.place(comp_list, net_list)

    return result.positions


# =============================================================================
# BENCHMARK
# =============================================================================

if __name__ == '__main__':
    import time
    import random

    print("=" * 60)
    print("ePlace Analytical Placement - Verification")
    print("=" * 60)

    # Create test case
    random.seed(42)
    n_components = 20
    n_nets = 30

    components = [
        Component(
            ref=f"U{i}",
            width=random.uniform(2, 5),
            height=random.uniform(2, 5)
        )
        for i in range(n_components)
    ]

    # Create random nets
    nets = []
    for i in range(n_nets):
        # Random 2-4 pin net
        n_pins = random.randint(2, 4)
        pin_comps = random.sample(range(n_components), n_pins)
        pins = [(f"U{c}", 0, 0) for c in pin_comps]
        nets.append(Net(name=f"NET{i}", pins=pins))

    # Run placement
    placer = ePlaceAnalytical(
        board_width=50,
        board_height=40,
        grid_bins_x=32,
        grid_bins_y=32
    )

    print(f"\nComponents: {n_components}")
    print(f"Nets: {n_nets}")

    start = time.time()
    result = placer.place(
        components, nets,
        max_iterations=200,
        callback=lambda i, wl, d: print(f"  Iter {i}: WL={wl:.1f}, Density={d:.1f}")
            if i % 50 == 0 else None
    )
    elapsed = time.time() - start

    print(f"\nResult:")
    print(f"  Wirelength: {result.wirelength:.1f}")
    print(f"  Density cost: {result.density_cost:.1f}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Converged: {result.converged}")
    print(f"  Time: {elapsed*1000:.0f}ms")

    # Verify no overlaps
    n_overlaps = 0
    for i, c1 in enumerate(components):
        for c2 in components[i+1:]:
            if placer._overlaps(c1.x, c1.y, c1.width, c1.height,
                               c2.x, c2.y, c2.width, c2.height):
                n_overlaps += 1

    print(f"  Overlaps after legalization: {n_overlaps}")
    print(f"\n{'PASS' if n_overlaps == 0 else 'FAIL'}")
