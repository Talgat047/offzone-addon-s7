# -*- coding: utf-8 -*-
"""Render an EasyEDA PCB json to PNG: outline, silkscreen, pads, tracks, vias."""
import io
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

D = "#@$"
COL = {1: "#d84040", 2: "#3a6fd8", 11: "#9a9a9a"}


def poly_from_path(d):
    """SOLIDREGION carries either an SVG path or a bare list of coordinates."""
    toks = d.replace(",", " ").split()
    if not toks:
        return []
    if not any(t.isalpha() for t in toks):
        n = [float(v) for v in toks]
        return [list(zip(n[0::2], n[1::2]))]
    pts, cur, i = [], [], 0
    while i < len(toks):
        t = toks[i]
        if t in ("M", "L"):
            cur.append((float(toks[i + 1]), float(toks[i + 2])))
            i += 3
        elif t in ("Z", "z"):
            if cur:
                pts.append(cur)
            cur = []
            i += 1
        else:
            i += 1
    if cur:
        pts.append(cur)
    return pts


def main(path, out, title=""):
    doc = json.load(io.open(path, encoding="utf-8"))
    ds = doc["schematics"][0]["dataStr"]
    shapes = ds["shape"]

    fig, ax = plt.subplots(figsize=(13, 13), dpi=190)

    for s in shapes:
        f = s.split("~")
        if f[0] == "SOLIDREGION" and f[1] == "3":
            for poly in poly_from_path(f[3]):
                ax.add_patch(Polygon(poly, closed=True, facecolor="#e8dcc0",
                                     edgecolor="none", zorder=1))
    for s in shapes:
        f = s.split("~")
        if f[0] == "TRACK" and f[2] == "10":
            n = [float(v) for v in f[4].split()]
            ax.plot(n[0::2], n[1::2], color="#111111", lw=1.4, zorder=6)

    for s in shapes:
        f = s.split("~")
        if f[0] == "TRACK" and f[2] in ("1", "2"):
            n = [float(v) for v in f[4].split()]
            ax.plot(n[0::2], n[1::2], color=COL[int(f[2])], lw=1.5,
                    solid_capstyle="round", zorder=3, alpha=0.95)
        elif f[0] == "VIA":
            ax.add_patch(Circle((float(f[1]), float(f[2])), float(f[3]) / 2,
                                facecolor="#2a2a2a", edgecolor="none", zorder=5))
            ax.add_patch(Circle((float(f[1]), float(f[2])), float(f[5]),
                                facecolor="white", edgecolor="none", zorder=5))

    for s in shapes:
        if not s.startswith("LIB~"):
            continue
        kv = s.split(D)[0].split("~")[3].split("`")
        attrs = dict(zip(kv[0::2], kv[1::2]))
        for sub in s.split(D)[1:]:
            g = sub.split("~")
            if g[0] != "PAD":
                continue
            lay = int(g[6])
            if g[10].strip() and g[1] == "RECT":
                n = [float(v) for v in g[10].split()]
                ax.add_patch(Polygon(list(zip(n[0::2], n[1::2])), closed=True,
                                     facecolor=COL.get(lay, "#888"),
                                     edgecolor="none", zorder=4))
            else:
                ax.add_patch(Circle((float(g[2]), float(g[3])),
                                    float(g[4]) / 2, facecolor=COL.get(lay, "#888"),
                                    edgecolor="none", zorder=4))
                if float(g[9]) > 0:
                    ax.add_patch(Circle((float(g[2]), float(g[3])), float(g[9]),
                                        facecolor="white", edgecolor="none", zorder=5))
        hx, hy = float(s.split("~")[1]), float(s.split("~")[2])
        ax.text(hx, hy - 6, attrs.get("Prefix", ""), fontsize=5.0, color="#000",
                ha="center", va="bottom", zorder=7,
                bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.5))

    bb = ds.get("BBox") or {}
    x0 = bb.get("x", 3870) - 10
    y0 = bb.get("y", 2870) - 10
    ax.set_xlim(x0, x0 + bb.get("width", 260) + 20)
    ax.set_ylim(y0 + bb.get("height", 260) + 20, y0)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, facecolor="white")
    print("rendered", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
