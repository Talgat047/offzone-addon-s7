# -*- coding: utf-8 -*-
"""
Take the placed-but-unrouted plane board (plane_easyeda.json), bring it in line
with the screenshot circuit and route it.

  1. Q1: KT-13 (КТ315, pads Э-К-Б in a row) -> SOT-23 for КТ3153А9, whose
     datasheet pin order is 1 = collector, 2 = base, 3 = emitter.
  2. LED1 and LED3 (one per wingtip) move off LEDDRV onto VCC so they burn
     steady; LED2, LED4 and LED5 keep blinking with the timer.
  3. Route every net on two layers with vias.

Units are EasyEDA's: 1 unit = 10 mil = 0.254 mm, y grows downwards.
"""
import heapq
import io
import json
import math
import os
import sys

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import binary_dilation, binary_erosion

D = "#@$"
MM = 0.254                      # mm per unit

GRID = 1.0                      # routing grid, units
TRACK_W = 1.0                   # 0.254 mm ~ 10 mil
CLEAR = 1.0                     # 0.254 mm ~ 10 mil
VIA_D = 2.4                     # 0.61 mm
VIA_HOLE_R = 0.6                # 0.3 mm drill
EDGE_MARGIN = 2.0               # keep copper 0.5 mm inside the outline

TOP, BOT = 0, 1                 # grid layer index
EE_LAYER = {TOP: 1, BOT: 2}

_gid = [20000]


def gid():
    _gid[0] += 1
    return "gge%d" % _gid[0]


def rot(dx, dy, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return dx * c - dy * s, dx * s + dy * c


def f3(v):
    return "%.3f" % v


# --------------------------------------------------------------- board I/O --

def load(path):
    doc = json.load(io.open(path, encoding="utf-8"))
    return doc, doc["schematics"][0]["dataStr"]


def lib_parts(shapes):
    """Split every LIB shape into (index, header fields, sub-shapes, attrs)."""
    out = []
    for i, s in enumerate(shapes):
        if not s.startswith("LIB~"):
            continue
        parts = s.split(D)
        hdr = parts[0].split("~")
        kv = hdr[3].split("`")
        attrs = dict(zip(kv[0::2], kv[1::2]))
        out.append({"i": i, "hdr": hdr, "subs": parts[1:], "attrs": attrs,
                    "ref": attrs.get("Prefix", "?")})
    return out


def pad_info(sub):
    f = sub.split("~")
    return {
        "shape": f[1], "x": float(f[2]), "y": float(f[3]),
        "w": float(f[4]), "h": float(f[5]), "layer": int(f[6]),
        "net": f[7], "num": f[8],
        "pts": [float(v) for v in f[10].split()] if f[10].strip() else [],
    }


def make_pad(num, cx, cy, w, h, layer, net, rotation):
    """Same field order the project already uses for its pads."""
    hw, hh = w / 2.0, h / 2.0
    pts = []
    for dx, dy in ((-hw, hh), (hw, hh), (hw, -hh), (-hw, -hh)):
        rx, ry = rot(dx, dy, rotation)
        pts += [f3(cx + rx), f3(cy + ry)]
    return "~".join(["PAD", "RECT", f3(cx), f3(cy), f3(w), f3(h), str(layer),
                     net, str(num), "0.000", " ".join(pts),
                     "%.1f" % (rotation % 360), gid(), "0", "", "Y", "0"])


# ------------------------------------------------------------ modifications --

# SOT-23 land pattern, offsets from the footprint origin in units.
# Two pads on one side, one opposite; 1.9 mm between pads 1 and 2.
SOT23 = {1: (4.8625, 3.74), 2: (4.8625, -3.74), 3: (-4.8625, 0.0)}
SOT23_W, SOT23_H = 4.2126, 2.3622

# КТ3153А9: 1 = коллектор, 2 = база, 3 = эмиттер (Далекс / Интеграл datasheets)
Q1_NETS = {1: "LEDDRV", 2: "BASE", 3: "GND"}

# one LED per wingtip goes steady, the rest keep blinking
STEADY_RES = ("R5", "R7")

# Both wingtip pairs were placed 2.42 mm apart, which leaves the facing 0805
# pads touching: LED1.2(GND)-LED2.1(A2) 0.068 mm and LED3.1(A3)-LED4.2(GND)
# 0.004 mm. That shorts LED2 and LED3 to ground. Push the inner LED of each
# pair further in until the gap is manufacturable.
LED_PAIRS = (("LED1", "LED2"), ("LED3", "LED4"))
MIN_PAD_GAP = 1.20              # 0.30 mm


def _seg_dist(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - (ax + dx * t), py - (ay + dy * t))


def poly_gap(pa, pb):
    """Smallest distance between two convex quads given as flat point lists."""
    A = list(zip(pa[0::2], pa[1::2]))
    B = list(zip(pb[0::2], pb[1::2]))
    best = 1e9
    for i in range(len(A)):
        for j in range(len(B)):
            best = min(best, _seg_dist(A[i], B[j], B[(j + 1) % len(B)]))
            best = min(best, _seg_dist(B[j], A[i], A[(i + 1) % len(A)]))
    return best


def translate_lib(part, dx, dy):
    """Shift a footprint: header, pads (centre and corners) and its silk text."""
    part["hdr"][1] = f3(float(part["hdr"][1]) + dx)
    part["hdr"][2] = f3(float(part["hdr"][2]) + dy)
    subs = []
    for s in part["subs"]:
        f = s.split("~")
        if f[0] == "PAD":
            f[2] = f3(float(f[2]) + dx)
            f[3] = f3(float(f[3]) + dy)
            if f[10].strip():
                n = [float(v) for v in f[10].split()]
                for i in range(0, len(n), 2):
                    n[i] += dx
                    n[i + 1] += dy
                f[10] = " ".join(f3(v) for v in n)
            s = "~".join(f)
        elif f[0] == "TEXT":
            f[2] = f3(float(f[2]) + dx)
            f[3] = f3(float(f[3]) + dy)
            s = "~".join(f)
        subs.append(s)
    part["subs"] = subs


def pads_of(part):
    out = []
    for s in part["subs"]:
        if s.startswith("PAD~"):
            out.append(pad_info(s))
    return out


def spread_led_pairs(shapes, parts):
    notes = []
    by_ref = {p["ref"]: p for p in parts}
    for outer, inner in LED_PAIRS:
        a, b = by_ref[outer], by_ref[inner]
        ax, ay = float(a["hdr"][1]), float(a["hdr"][2])
        bx, by = float(b["hdr"][1]), float(b["hdr"][2])
        d = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / d, (by - ay) / d      # points from outer to inner
        step, moved = 0.1, 0.0
        while moved < 12.0:
            gap = min(poly_gap(pa["pts"], pb["pts"])
                      for pa in pads_of(a) for pb in pads_of(b)
                      if pa["net"] != pb["net"])
            if gap >= MIN_PAD_GAP:
                break
            translate_lib(b, ux * step, uy * step)
            moved += step
        shapes[b["i"]] = D.join(["~".join(b["hdr"])] + b["subs"])
        gap = min(poly_gap(pa["pts"], pb["pts"])
                  for pa in pads_of(a) for pb in pads_of(b)
                  if pa["net"] != pb["net"])
        notes.append("%s moved %.2f mm away from %s: pad gap %.3f -> %.3f mm"
                     % (inner, moved * MM, outer,
                        0.068 if outer == "LED1" else 0.004, gap * MM))
    return notes


def retarget(shapes, parts):
    changed = []

    q1 = [p for p in parts if p["ref"] == "Q1"][0]
    ox, oy = float(q1["hdr"][1]), float(q1["hdr"][2])
    deg = float(q1["hdr"][4] or 0)
    keep = [s for s in q1["subs"] if not s.startswith("PAD~")]
    new_pads = []
    for num, (dx, dy) in sorted(SOT23.items()):
        rx, ry = rot(dx, dy, deg)
        new_pads.append(make_pad(num, ox + rx, oy + ry, SOT23_W, SOT23_H,
                                 2, Q1_NETS[num], deg))
    q1["attrs"]["package"] = "SOT-23"
    kv = "".join("%s`%s`" % (k, v) for k, v in q1["attrs"].items())
    q1["hdr"][3] = kv
    shapes[q1["i"]] = D.join(["~".join(q1["hdr"])] + new_pads + keep)
    changed.append("Q1: KT-13-SMD -> SOT-23, pads 1=LEDDRV(K) 2=BASE(B) 3=GND(E)")

    for ref in STEADY_RES:
        p = [q for q in parts if q["ref"] == ref][0]
        subs = []
        for s in p["subs"]:
            if s.startswith("PAD~") and s.split("~")[7] == "LEDDRV":
                f = s.split("~")
                f[7] = "VCC"
                s = "~".join(f)
            subs.append(s)
        shapes[p["i"]] = D.join(["~".join(p["hdr"])] + subs)
        changed.append("%s: LEDDRV -> VCC (its LED now burns steady)" % ref)
    return changed


# ------------------------------------------------------------------ routing --

class Board(object):
    def __init__(self, shapes):
        self.shapes = shapes
        outlines = []
        for s in shapes:
            f = s.split("~")
            if f[0] == "TRACK" and f[2] == "10":
                n = [float(v) for v in f[4].split()]
                outlines.append(list(zip(n[0::2], n[1::2])))
        outlines.sort(key=lambda p: -_area(p))
        self.outer, self.holes = outlines[0], outlines[1:]

        xs = [p[0] for p in self.outer]
        ys = [p[1] for p in self.outer]
        self.x0, self.x1 = min(xs) - 2, max(xs) + 2
        self.y0, self.y1 = min(ys) - 2, max(ys) + 2
        self.nx = int((self.x1 - self.x0) / GRID) + 1
        self.ny = int((self.y1 - self.y0) / GRID) + 1

        gx = self.x0 + np.arange(self.nx) * GRID
        gy = self.y0 + np.arange(self.ny) * GRID
        self.gx, self.gy = gx, gy
        XX, YY = np.meshgrid(gx, gy)
        pts = np.column_stack([XX.ravel(), YY.ravel()])

        # Erode the rasterised outline instead of Path(radius=...): the sign of
        # that radius follows the polygon winding, and on this outline it grew
        # the board instead of shrinking it, letting tracks run off the edge.
        margin = int(math.ceil(EDGE_MARGIN / GRID))
        inside = Path(self.outer).contains_points(pts).reshape(self.ny, self.nx)
        inside = binary_erosion(inside, structure=_disk(margin))
        for h in self.holes:
            hm = Path(h).contains_points(pts).reshape(self.ny, self.nx)
            inside &= ~binary_dilation(hm, structure=_disk(margin))
        self.inside = inside

        self.pads = []
        for p in lib_parts(shapes):
            for sub in p["subs"]:
                if sub.startswith("PAD~"):
                    d = pad_info(sub)
                    d["ref"] = p["ref"]
                    self.pads.append(d)

        # copper = pads + whatever the router lays down; routedcu is the router's
        # share alone, so a clearance report can tell the two apart
        self.copper = {TOP: {}, BOT: {}}
        self.padcu = {TOP: {}, BOT: {}}
        self.routedcu = {TOP: {}, BOT: {}}
        for d in self.pads:
            if not d["net"]:
                continue
            m = self._raster(d)
            for lay in self._pad_layers(d):
                self.copper[lay].setdefault(d["net"], self._empty())
                self.copper[lay][d["net"]] |= m
                self.padcu[lay].setdefault(d["net"], self._empty())
                self.padcu[lay][d["net"]] |= m
        self.tracks = []
        self.vias = []

    def _empty(self):
        return np.zeros((self.ny, self.nx), dtype=bool)

    def _pad_layers(self, d):
        if d["layer"] == 11:
            return (TOP, BOT)
        return (TOP,) if d["layer"] == 1 else (BOT,)

    def _raster(self, d):
        XX, YY = np.meshgrid(self.gx, self.gy)
        if d["shape"] == "ELLIPSE" or not d["pts"]:
            r = max(d["w"], d["h"]) / 2.0
            return ((XX - d["x"]) ** 2 + (YY - d["y"]) ** 2) <= r * r
        poly = list(zip(d["pts"][0::2], d["pts"][1::2]))
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        return Path(poly).contains_points(pts, radius=0.35).reshape(self.ny, self.nx)

    def cell(self, x, y):
        return (int(round((x - self.x0) / GRID)), int(round((y - self.y0) / GRID)))

    def coord(self, ix, iy):
        return self.x0 + ix * GRID, self.y0 + iy * GRID

    # -- obstacle map for one net -------------------------------------------
    def blocked_for(self, net):
        r = int(math.ceil((CLEAR + TRACK_W / 2.0) / GRID))
        st = _disk(r)
        out = {}
        for lay in (TOP, BOT):
            foreign = self._empty()
            for n, m in self.copper[lay].items():
                if n != net:
                    foreign |= m
            grown = binary_dilation(foreign, structure=st) if foreign.any() else foreign
            blocked = grown | ~self.inside
            # our own copper is legitimately there whatever the neighbours do,
            # otherwise a pad sitting inside a foreign halo becomes unreachable
            own = self.copper[lay].get(net)
            if own is not None:
                blocked &= ~own
            out[lay] = blocked
        return out

    def add_copper(self, net, lay, mask):
        self.copper[lay].setdefault(net, self._empty())
        self.copper[lay][net] |= mask
        self.routedcu[lay].setdefault(net, self._empty())
        self.routedcu[lay][net] |= mask


def _area(poly):
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


ORTHO = [(1, 0, 10), (-1, 0, 10), (0, 1, 10), (0, -1, 10)]
DIAG = [(1, 1, 14), (1, -1, 14), (-1, 1, 14), (-1, -1, 14)]
VIA_COST = 90


def route_net(bd, net, verbose=True):
    """Connect every pad of `net` with a Dijkstra search over both layers."""
    groups = []
    seen = set()
    for d in bd.pads:
        if d["net"] != net:
            continue
        key = (d["ref"], d["num"])
        if key in seen:
            continue
        seen.add(key)
        cells = set()
        m = bd._raster(d)
        idx = np.argwhere(m)
        for iy, ix in idx:
            for lay in bd._pad_layers(d):
                cells.add((lay, int(iy), int(ix)))
        if cells:
            groups.append(cells)
    if len(groups) < 2:
        return True, 0

    blocked = bd.blocked_for(net)
    tree = set(groups[0])
    rest = groups[1:]
    laid = 0
    while rest:
        best = None
        for gi, g in enumerate(rest):
            path = _search(bd, tree, g, blocked)
            if path is not None:
                best = (gi, path)
                break
        if best is None:
            return False, laid
        gi, path = best
        _commit(bd, net, path, blocked)
        tree |= set(path) | set(rest[gi])
        rest.pop(gi)
        laid += 1
    return True, laid


def _search(bd, starts, goals, blocked):
    INF = 1 << 30
    dist = {}
    prev = {}
    heap = []
    for c in starts:
        lay, iy, ix = c
        if blocked[lay][iy, ix]:
            continue
        dist[c] = 0
        heapq.heappush(heap, (0, c))
    goalset = set(goals)
    while heap:
        d, c = heapq.heappop(heap)
        if d > dist.get(c, INF):
            continue
        if c in goalset:
            path = [c]
            while c in prev:
                c = prev[c]
                path.append(c)
            return path[::-1]
        lay, iy, ix = c
        for dx, dy, w in ORTHO + DIAG:
            ny_, nx_ = iy + dy, ix + dx
            if not (0 <= ny_ < bd.ny and 0 <= nx_ < bd.nx):
                continue
            n = (lay, ny_, nx_)
            if blocked[lay][ny_, nx_]:
                continue
            nd = d + w
            if nd < dist.get(n, INF):
                dist[n] = nd
                prev[n] = c
                heapq.heappush(heap, (nd, n))
        other = BOT if lay == TOP else TOP
        n = (other, iy, ix)
        if not blocked[other][iy, ix]:
            nd = d + VIA_COST
            if nd < dist.get(n, INF):
                dist[n] = nd
                prev[n] = c
                heapq.heappush(heap, (nd, n))
    return None


def _commit(bd, net, path, blocked):
    """Turn a cell path into TRACK/VIA shapes and stamp it into the copper map."""
    runs = []
    cur = [path[0]]
    for c in path[1:]:
        if c[0] != cur[-1][0]:
            runs.append(cur)
            bd.vias.append((cur[-1][2], cur[-1][1], net))
            cur = [c]
        else:
            cur.append(c)
    runs.append(cur)

    for run in runs:
        lay = run[0][0]
        # stamp copper for every run, even a one-cell stub, or the net would
        # come out split; only runs with real length become a TRACK
        m = bd._empty()
        for _, iy, ix in run:
            m[iy, ix] = True
        bd.add_copper(net, lay, m)
        if len(run) >= 2:
            pts = _simplify([bd.coord(ix, iy) for _, iy, ix in run])
            bd.tracks.append((lay, net, pts))
    if len(runs) > 1:
        for ix, iy, _n in bd.vias[-(len(runs) - 1):]:
            for lay in (TOP, BOT):
                m = bd._empty()
                m[iy, ix] = True
                bd.add_copper(net, lay, m)


def _simplify(pts):
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        ax, ay = out[-1]
        bx, by = pts[i]
        cx, cy = pts[i + 1]
        if (bx - ax) * (cy - by) != (by - ay) * (cx - bx):
            out.append(pts[i])
    out.append(pts[-1])
    return out


# ------------------------------------------------------------------- output --

def emit(bd, shapes):
    for lay, net, pts in bd.tracks:
        flat = " ".join("%s %s" % (f3(x), f3(y)) for x, y in pts)
        shapes.append("~".join(["TRACK", f3(TRACK_W), str(EE_LAYER[lay]),
                                net, flat, gid(), "0"]))
    for ix, iy, net in bd.vias:
        x, y = bd.coord(ix, iy)
        shapes.append("~".join(["VIA", f3(x), f3(y), f3(VIA_D), net,
                                f3(VIA_HOLE_R), gid(), "0"]))


def validate(bd):
    """Rebuild connectivity straight from the emitted copper and check it."""
    # cells that tie the two layers together: vias, and through-hole pads,
    # which span both layers by construction
    via_cells = {}
    for ix, iy, net in bd.vias:
        via_cells.setdefault(net, set()).add((iy, ix))
    for d in bd.pads:
        if d["net"] and d["layer"] == 11:
            for iy, ix in np.argwhere(bd._raster(d)):
                via_cells.setdefault(d["net"], set()).add((int(iy), int(ix)))

    problems = []
    nets = sorted({d["net"] for d in bd.pads if d["net"]})
    for net in nets:
        cells = set()
        for lay in (TOP, BOT):
            m = bd.copper[lay].get(net)
            if m is None:
                continue
            for iy, ix in np.argwhere(m):
                cells.add((lay, int(iy), int(ix)))
        if not cells:
            continue
        seen = set()
        stack = [next(iter(cells))]
        vias = via_cells.get(net, set())
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            lay, iy, ix = c
            for dx, dy, _ in ORTHO + DIAG:
                n = (lay, iy + dy, ix + dx)
                if n in cells and n not in seen:
                    stack.append(n)
            if (iy, ix) in vias:
                n = (BOT if lay == TOP else TOP, iy, ix)
                if n in cells and n not in seen:
                    stack.append(n)
        if len(seen) != len(cells):
            islands = 1
            rest = cells - seen
            while rest:
                islands += 1
                st = [next(iter(rest))]
                grp = set()
                while st:
                    c = st.pop()
                    if c in grp:
                        continue
                    grp.add(c)
                    lay, iy, ix = c
                    for dx, dy, _ in ORTHO + DIAG:
                        n = (lay, iy + dy, ix + dx)
                        if n in rest and n not in grp:
                            st.append(n)
                    if (iy, ix) in vias:
                        n = (BOT if lay == TOP else TOP, iy, ix)
                        if n in rest and n not in grp:
                            st.append(n)
                rest -= grp
            problems.append("net %s is split into %d islands" % (net, islands))

    # pad-to-pad clearance, measured on the real polygons rather than the grid
    for i in range(len(bd.pads)):
        a = bd.pads[i]
        if not a["pts"]:
            continue
        for j in range(i + 1, len(bd.pads)):
            b = bd.pads[j]
            if not b["pts"] or a["net"] == b["net"] or a["ref"] == b["ref"]:
                continue
            if not (a["layer"] == b["layer"] or 11 in (a["layer"], b["layer"])):
                continue
            if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) > 30:
                continue
            g = poly_gap(a["pts"], b["pts"])
            if g < MIN_PAD_GAP:
                problems.append("pad gap %.3f mm between %s.%s(%s) and %s.%s(%s)"
                                % (g * MM, a["ref"], a["num"], a["net"],
                                   b["ref"], b["num"], b["net"]))

    # Clearance is only checked against copper the router put down. Pad-to-pad
    # spacing is a property of the footprints, not of the routing, and on a
    # 1.27 mm-pitch SOIC it sits below this grid's resolution anyway.
    st = _disk(int(math.ceil(CLEAR / GRID)))
    for lay in (TOP, BOT):
        for na, ma in bd.routedcu[lay].items():
            # a track crossing its own pad is pad copper, not a routing fault
            pad = bd.padcu[lay].get(na)
            if pad is not None:
                ma = ma & ~pad
            if not ma.any():
                continue
            grown = binary_dilation(ma, structure=st)
            for nb, mb in bd.copper[lay].items():
                if nb == na or not mb.any():
                    continue
                hit = grown & mb
                if hit.any():
                    iy, ix = np.argwhere(hit)[0]
                    x, y = bd.coord(int(ix), int(iy))
                    problems.append(
                        "clearance: %s track too close to %s on layer %d at "
                        "(%.1f, %.1f) mm" % (na, nb, EE_LAYER[lay],
                                             (x - bd.x0) * MM, (y - bd.y0) * MM))
    return problems


def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    doc, ds = load(src)
    shapes = ds["shape"]

    parts = lib_parts(shapes)
    for line in retarget(shapes, parts):
        print("  ", line)
    for line in spread_led_pairs(shapes, parts):
        print("  ", line)

    bd = Board(shapes)
    print("board %.1f x %.1f mm, grid %dx%d, pads %d"
          % ((bd.x1 - bd.x0) * MM, (bd.y1 - bd.y0) * MM, bd.nx, bd.ny, len(bd.pads)))

    nets = {}
    for d in bd.pads:
        if d["net"]:
            nets.setdefault(d["net"], 0)
            nets[d["net"]] += 1
    order = sorted(nets, key=lambda n: nets[n])
    print("\nrouting %d nets" % len(order))
    failed = []
    for net in order:
        ok, laid = route_net(bd, net)
        print("   %-8s pads=%-3d %s" % (net, nets[net], "ok" if ok else "FAILED"))
        if not ok:
            failed.append(net)

    emit(bd, shapes)
    print("\ntracks=%d vias=%d" % (len(bd.tracks), len(bd.vias)))

    problems = validate(bd)
    print("\n--- CHECKS ---")
    if failed:
        problems.insert(0, "unrouted nets: %s" % ", ".join(failed))
    if problems:
        for p in problems:
            print("   FAIL:", p)
    else:
        print("   every net is one connected island; no clearance violations")

    with io.open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("\nwritten", dst)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
