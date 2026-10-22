"""Restricted LaTeX arithmetic parser and numerical 2D density sampler.

No eval(), exec(), SymPy parsing or arbitrary function calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from region import Region

INITIAL = r"\frac{1}{2\pi}e^{-\frac{x^2+y^2}{2}}"
TARGET = r"\frac{1}{2\pi}e^{-\frac{(x-2)^2+(y-1)^2}{2}}"
COMPACT = r"\max(0,1-x^2-y^2)"
MIXTURE = (
    r"0.5e^{-\frac{(x+2)^2+y^2}{2}}"
    r"+0.5e^{-\frac{(x-2)^2+y^2}{2}}"
)

_TOKEN = re.compile(r"\\frac|\\exp|\\max|\\pi|(?:\d+(?:\.\d*)?|\.\d+)|[xy e]|[+*/^(),{}-]".replace("[xy e]", "[xye]"))
_START = {"NUM", "x", "y", "e", "\\pi", "\\frac", "\\exp", "\\max", "(", "{"}


class DensitySyntaxError(ValueError):
    """Invalid or unsupported formula."""


class Parser:
    def __init__(self, formula: str):
        if len(formula) > 500:
            raise DensitySyntaxError("Formula too long (maximum 500 characters).")
        src = formula.replace(r"\left", "").replace(r"\right", "")
        src = re.sub(r"\\(?:,|;|!|quad|qquad)", "", src)
        src = src.replace(r"\cdot", "*").replace(r"\times", "*")
        src = re.sub(r"\s+", "", src)
        self.tokens: list[tuple[str, str]] = []
        i = 0
        while i < len(src):
            match = _TOKEN.match(src, i)
            if match is None:
                raise DensitySyntaxError(f"Unsupported LaTeX near: {src[i:i+24]!r}")
            t = match.group()
            self.tokens.append(("NUM" if t[0].isdigit() or t[0] == "." else t, t))
            i = match.end()
        self.tokens.append(("EOF", ""))
        self.i = 0

    def kind(self):
        return self.tokens[self.i][0]

    def eat(self, kind):
        if self.kind() != kind:
            raise DensitySyntaxError(f"Expected {kind!r}, got {self.tokens[self.i][1]!r}")
        value = self.tokens[self.i][1]
        self.i += 1
        return value

    def parse(self):
        result = self.expr()
        self.eat("EOF")
        return result

    def expr(self):
        node = self.term()
        while self.kind() in {"+", "-"}:
            op = self.kind()
            self.eat(op)
            node = (op, node, self.term())
        return node

    def term(self):
        node = self.unary()
        while True:
            op = self.kind()
            if op in {"*", "/"}:
                self.eat(op)
                node = (op, node, self.unary())
            elif op in _START:
                node = ("*", node, self.unary())
            else:
                break
        return node

    def unary(self):
        if self.kind() == "+":
            self.eat("+")
            return self.unary()
        if self.kind() == "-":
            self.eat("-")
            return ("neg", self.unary())
        return self.power()

    def power(self):
        node = self.atom()
        if self.kind() == "^":
            self.eat("^")
            node = ("^", node, self.unary())
        return node

    def group(self):
        op = self.kind()
        if op not in {"(", "{"}:
            raise DensitySyntaxError("Expected a parenthesized or braced expression")
        self.eat(op)
        node = self.expr()
        self.eat(")" if op == "(" else "}")
        return node

    def atom(self):
        kind = self.kind()
        if kind == "NUM":
            return ("const", float(self.eat("NUM")))
        if kind in {"x", "y", "e", "\\pi"}:
            self.eat(kind)
            return ("var", kind)
        if kind in {"(", "{"}:
            return self.group()
        if kind == "\\frac":
            self.eat(kind)
            return ("/", self.group(), self.group())
        if kind == "\\exp":
            self.eat(kind)
            return ("exp", self.group())
        if kind == "\\max":
            self.eat(kind)
            opener = self.kind()
            if opener not in {"(", "{"}:
                raise DensitySyntaxError(r"Use \max(a,b)")
            self.eat(opener)
            a = self.expr()
            self.eat(",")
            b = self.expr()
            self.eat(")" if opener == "(" else "}")
            return ("max", a, b)
        raise DensitySyntaxError(f"Unexpected token {self.tokens[self.i][1]!r}")


def evaluate(tree, x: np.ndarray, y: np.ndarray):
    """Evaluate an AST on numpy arrays with limited operations."""
    kind = tree[0]
    if kind == "const":
        return tree[1]
    if kind == "var":
        return {"x": x, "y": y, "e": np.e, r"\pi": np.pi}[tree[1]]
    if kind == "neg":
        return -evaluate(tree[1], x, y)
    if kind == "exp":
        with np.errstate(over="ignore", invalid="ignore"):
            return np.exp(evaluate(tree[1], x, y))
    if kind == "max":
        return np.maximum(evaluate(tree[1], x, y), evaluate(tree[2], x, y))
    a = evaluate(tree[1], x, y)
    b = evaluate(tree[2], x, y)
    with np.errstate(all="ignore"):
        if kind == "+":
            return a + b
        if kind == "-":
            return a - b
        if kind == "*":
            return a * b
        if kind == "/":
            return a / b
        if kind == "^":
            return np.power(a, b)
    raise DensitySyntaxError(f"Unsupported operation {kind}")


@dataclass
class GridDensity:
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    mass_before_normalization: float
    region: Region | None = None


_INDICATOR = re.compile(r"(?:\\mathbf\s*\{1\}|\\mathbb\s*\{1\}|\\chi)\s*_\s*(?:\{A\}|A)\s*$")


def split_indicator(formula: str) -> tuple[str, bool]:
    """Split one trailing characteristic-function factor from a formula."""
    src = formula.strip()
    match = _INDICATOR.search(src)
    if match:
        src = src[:match.start()].strip()
        src = re.sub(r"(?:\\cdot|\\times|\*)\s*$", "", src).strip()
        if not src:
            src = "1"
        if _INDICATOR.search(src) or r"\mathbf" in src or r"\mathbb" in src or r"\chi" in src:
            raise DensitySyntaxError("Only one trailing indicator 1_A is supported")
        return src, True
    if r"\mathbf" in src or r"\mathbb" in src or r"\chi" in src:
        raise DensitySyntaxError(r"Use a single trailing indicator \mathbf{1}_A and select region A")
    return src, False


def grid_density(formula: str, bounds: tuple[float, float, float, float], resolution: int = 180,
                 region: Region | None = None) -> GridDensity:
    xmin, xmax, ymin, ymax = bounds
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("Invalid domain bounds")
    if resolution < 2:
        raise ValueError("Grid resolution must be at least 2")
    expression, has_indicator = split_indicator(formula)
    if has_indicator and region is None:
        raise ValueError("Formula contains 1_A: select a geometric region A")
    tree = Parser(expression).parse()
    # Cell centers; each grid point represents an equally sized cell.
    xs = xmin + (np.arange(resolution) + 0.5) * (xmax - xmin) / resolution
    ys = ymin + (np.arange(resolution) + 0.5) * (ymax - ymin) / resolution
    xx, yy = np.meshgrid(xs, ys)
    raw = np.asarray(evaluate(tree, xx, yy), dtype=float)
    raw = np.broadcast_to(raw, xx.shape).copy()
    if region is not None:
        raw = np.where(region.contains(xx, yy), raw, 0.0)
    if not np.all(np.isfinite(raw)):
        raise ValueError("Density is nonfinite on the selected support")
    if np.min(raw) < -1e-12 * float(np.max(np.abs(raw))):
        raise ValueError("Density takes negative values on the selected support")
    raw = np.maximum(raw, 0)
    area = (xmax - xmin) * (ymax - ymin) / resolution**2
    mass = float(raw.sum() * area)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Density has zero or invalid mass on the domain")
    return GridDensity(xs, ys, raw / mass, mass, region)


def sample_density(grid: GridDensity, count: int, seed: int) -> np.ndarray:
    if count < 1:
        raise ValueError("Sample count must be positive")
    rng = np.random.default_rng(seed)
    weights = grid.values.ravel()
    ids = rng.choice(weights.size, size=count, p=weights / weights.sum())
    rows, cols = np.divmod(ids, len(grid.x))
    dx = grid.x[1] - grid.x[0]
    dy = grid.y[1] - grid.y[0]
    points = np.column_stack((
        grid.x[cols] + rng.uniform(-dx / 2, dx / 2, count),
        grid.y[rows] + rng.uniform(-dy / 2, dy / 2, count),
    ))
    if grid.region is not None:
        invalid = ~grid.region.contains(points[:, 0], points[:, 1])
        for _ in range(200):
            if not np.any(invalid):
                break
            bad_count = int(invalid.sum())
            points[invalid, 0] = grid.x[cols[invalid]] + rng.uniform(-dx / 2, dx / 2, bad_count)
            points[invalid, 1] = grid.y[rows[invalid]] + rng.uniform(-dy / 2, dy / 2, bad_count)
            invalid = ~grid.region.contains(points[:, 0], points[:, 1])
        if np.any(invalid):
            raise ValueError("Mask boundary underresolved: increase sampling grid resolution")
    return points
