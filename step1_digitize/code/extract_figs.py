"""Extract the per-N data behind Fig. 2 and Fig. 5 of
Gomez Cadavid et al., "Scaling advantage with quantum-enhanced memetic tabu search for LABS",
Quantum Mach. Intell. 8:92 (2026), doi:10.1007/s42484-026-00433-1 (publisher PDF).

  Fig. 5 (page 8) is a VECTOR graphic -> values are read from the PDF drawing commands
         (precision ~1e-3 in log10 units, limited only by the PDF coordinates).
  Fig. 2 (page 6) is an embedded RASTER image (856x599 JPEG) -> the black median-of-medians
         markers (Q0.50) are located by connected components; the log10 TTS axis is calibrated from
         the 17 detected major/minor tick marks (1e6 major tick + 2e5..9e6 minor ticks).

usage:  python extract_figs.py PATH/TO/s42484-026-00433-1.pdf  [outdir]
needs:  pymupdf, numpy, scipy, pillow, matplotlib
writes: fig2_q50_digitized.csv, fig5_logratio_vector.csv, fig2_overlay.png, fig5_overlay.png,
        extraction_log.txt
"""
import sys, os, io, json
import numpy as np
import pymupdf as fitz
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path

PDF = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

doc = fitz.open(PDF)
NS = list(range(27, 38))

# ----------------------------------------------------------------------------------------------
# Fig. 5  (vector)  page index 7, upper-left panel
# ----------------------------------------------------------------------------------------------
p5 = doc[7]
REG5 = fitz.Rect(40, 50, 300, 235)
dr = [d for d in p5.get_drawings() if REG5.y0 <= d["rect"].y0 and d["rect"].y1 <= REG5.y1 and REG5.x0 <= d["rect"].x0 and d["rect"].x1 <= REG5.x1]

def seg_list(d):
    return [it for it in d["items"] if it[0] == "l"]

# axis frame: left spine x and bottom spine y
frame = [d for d in dr if d["type"] == "s" and d.get("color") == (0.0, 0.0, 0.0) and len(d["items"]) >= 3]
xs_left = min(min(it[1].x, it[2].x) for d in frame for it in d["items"])
y_bottom = max(max(it[1].y, it[2].y) for d in frame for it in d["items"])
# tick lines (single short black segments)
ticks = [d["items"][0] for d in dr if d["type"] == "fs" and len(d["items"]) == 1 and d["items"][0][0] == "l"]
yt = sorted([(it[1].y) for it in ticks if abs(it[1].y - it[2].y) < 1e-3])            # horizontal -> y ticks
xt = sorted([(it[1].x) for it in ticks if abs(it[1].x - it[2].x) < 1e-3])            # vertical   -> x ticks

# y tick labels from text spans left of the axis (join spans on the same baseline)
spans = []
for b in p5.get_text("dict", clip=fitz.Rect(64, 50, xs_left - 1, y_bottom + 2))["blocks"]:
    for l in b.get("lines", []):
        for s in l["spans"]:
            spans.append(((s["bbox"][1] + s["bbox"][3]) / 2, s["bbox"][0], s["text"]))
rows = {}
for yc, x0, t in spans:
    key = min(rows, key=lambda k: abs(k - yc)) if rows else None
    if key is None or abs(key - yc) > 3: rows[yc] = [(x0, t)]
    else: rows[key].append((x0, t))
ylab = []
for yc, parts in rows.items():
    txt = "".join(t for _, t in sorted(parts)).replace("−", "-").strip()
    try: ylab.append((yc, float(txt)))
    except ValueError: pass
pairs = []
for y in yt:
    yc, v = min(ylab, key=lambda r: abs(r[0] - y)); pairs.append((y, v))
A = np.array([[1, y] for y, _ in pairs]); bv = np.array([v for _, v in pairs])
c5, *_ = np.linalg.lstsq(A, bv, rcond=None)
res5 = np.abs(A @ c5 - bv).max()
y2D = lambda y: c5[0] + c5[1] * y
log(f"[Fig5] y-ticks {pairs}  ->  D = {c5[0]:.5f} + {c5[1]:.6f}*y_pt   (max resid {res5:.2e})")
# x ticks -> N
assert len(xt) == 11, xt
x2N = dict(zip(np.round(xt, 2), NS))
def nearest_N(x):
    i = int(np.argmin([abs(x - t) for t in xt])); return NS[i], abs(x - xt[i])
log(f"[Fig5] x-ticks (pt): {np.round(xt,2).tolist()}")

# median markers: black stroke, width ~1.22, short horizontal segments
med = {}
for d in dr:
    if d["type"] == "s" and d.get("color") == (0.0, 0.0, 0.0) and d.get("width") and abs(d["width"] - 1.219) < 0.05:
        for it in seg_list(d):
            if abs(it[1].y - it[2].y) < 1e-3:
                xc = (it[1].x + it[2].x) / 2; N, dx = nearest_N(xc)
                med[N] = y2D(it[1].y)
assert sorted(med) == NS, med

# violins: filled polygons; per N the smaller one is the KDE body, the larger one is the stroked outline
# the violin colour is the most common fill among large polygons (excludes glyph outlines)
from collections import Counter
cand = [d for d in dr if d["type"] == "f" and d.get("fill") and len(d["items"]) > 50]
vcol = Counter(tuple(round(c, 3) for c in d["fill"]) for d in cand).most_common(1)[0][0]
viol = [d for d in cand if tuple(round(c, 3) for c in d["fill"]) == vcol]
log(f"[Fig5] violin fill colour {vcol}, {len(viol)} polygons")
byN = {}
for d in viol:
    N, dx = nearest_N((d["rect"].x0 + d["rect"].x1) / 2)
    byN.setdefault(N, []).append(d)
def poly_pts(d):
    P = []
    for it in d["items"]:
        if it[0] == "l": P += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        elif it[0] == "c":   # cubic bezier: sample it
            p0, p1, p2, p3 = [np.array([q.x, q.y]) for q in it[1:5]]
            for t in np.linspace(0, 1, 9):
                P.append(tuple((1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3))
    return np.array(P)
rows5 = []
for N in NS:
    ds = sorted(byN[N], key=lambda d: d["rect"].height)
    body, outline = ds[0], ds[-1]
    # stroke half-width estimated from horizontal growth of outline vs body
    hw = ((body["rect"].x0 - outline["rect"].x0) + (outline["rect"].x1 - body["rect"].x1)) / 2
    Dmax = y2D(outline["rect"].y0 + hw); Dmin = y2D(outline["rect"].y1 - hw)
    # body area fraction below D=0 (KDE body only; tails beyond the body are not included)
    P = poly_pts(body); path = Path(P)
    xs = np.linspace(body["rect"].x0, body["rect"].x1, 400); ys = np.linspace(body["rect"].y0, body["rect"].y1, 1600)
    X, Y = np.meshgrid(xs, ys); inside = path.contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    y0pt = (0 - c5[0]) / c5[1]
    frac_neg = inside[ys > y0pt].sum() / inside.sum()
    # body mode (widest row)
    w = inside.sum(1); Dmode = y2D(ys[np.argmax(w)])
    rows5.append(dict(N=N, median_D=med[N], body_top_D=y2D(body["rect"].y0), body_bottom_D=y2D(body["rect"].y1),
                      body_mode_D=Dmode, min_D=Dmin, max_D=Dmax, body_frac_below0=frac_neg))
with open(os.path.join(OUT, "fig5_logratio_vector.csv"), "w") as f:
    f.write("# Fig.5 of QMI 8:92 (vector). D = log10(TTS_QE-MTS) - log10(TTS_MTS); per-N bootstrap distribution.\n")
    f.write("# median_D: black marker. body_*: KDE body of the violin. min_D/max_D: ends of the extrema line\n")
    f.write("# (stroke half-width removed). body_frac_below0: area fraction of the KDE body with D<0 (tails excluded).\n")
    keys = list(rows5[0])
    f.write(",".join(keys) + "\n")
    for r in rows5: f.write(",".join(f"{r[k]:.4f}" if k != "N" else str(r[k]) for k in keys) + "\n")
log("[Fig5] N, median D, body range, extrema, frac<0:")
for r in rows5:
    log(f"   {r['N']}  {r['median_D']:+.4f}  body [{r['body_bottom_D']:+.3f},{r['body_top_D']:+.3f}]  "
        f"ext [{r['min_D']:+.3f},{r['max_D']:+.3f}]  P_body(D<0)={r['body_frac_below0']:.3f}")

# overlay check for Fig.5
pix = p5.get_pixmap(clip=fitz.Rect(45, 55, 290, 230), dpi=300)
img5 = Image.open(io.BytesIO(pix.tobytes("png")))
sc = 300 / 72
fig, ax = plt.subplots(figsize=(7, 5)); ax.imshow(img5)
for r in rows5:
    xpt = xt[NS.index(r["N"])]; ypt = (r["median_D"] - c5[0]) / c5[1]
    ax.plot((xpt - 45) * sc, (ypt - 55) * sc, "r+", ms=14, mew=1.5)
ax.set_title("Fig.5 check: red + = extracted medians"); ax.axis("off")
fig.savefig(os.path.join(OUT, "fig5_overlay.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# Fig. 2  (raster)  page index 5
# ----------------------------------------------------------------------------------------------
p2 = doc[5]
imgs = p2.get_images(full=True)
xref = max(imgs, key=lambda im: im[2] * im[3])[0]
raw = doc.extract_image(xref)
img = Image.open(io.BytesIO(raw["image"])).convert("RGB")
a = np.asarray(img).astype(float); g = a.mean(2); H, W = g.shape
log(f"[Fig2] embedded image xref={xref} {W}x{H} {raw['ext']}")

def centers(profile, thr=128):
    idx = np.where(profile < thr)[0]
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1: cur.append(i)
        else: groups.append(cur); cur = [i]
    groups.append(cur)
    out = []
    for gr in groups:
        wgt = 255 - profile[gr]; out.append(float((np.array(gr) * wgt).sum() / wgt.sum()))
    return out

# left spine = first column that is dark over most of the height; bottom spine likewise
col_dark = (g < 100).sum(0); spine_x = int(np.where(col_dark > 0.8 * H)[0].min())   # left spine
row_dark = (g < 100).sum(1); spine_y = int(np.where(row_dark > 0.8 * W)[0].max())   # bottom spine (lowest full dark row)
minor = centers(g[:, spine_x - 4]); major = centers(g[:, 2])
assert len(major) == 1, major
y_major = major[0]
above = sorted([y for y in minor if y < y_major - 2], reverse=True)       # 2e6, 3e6, ... upwards
below = sorted([y for y in minor if y > y_major + 2])                      # 9e5, 8e5, ... downwards
tv = [(y_major, 1e6)] + [(y, (k + 2) * 1e6) for k, y in enumerate(above)] + [(y, (9 - k) * 1e5) for k, y in enumerate(below)]
Yt = np.array([y for y, _ in tv]); Lt = np.log10([v for _, v in tv])
c2, *_ = np.linalg.lstsq(np.c_[np.ones_like(Yt), Yt], Lt, rcond=None)
rms2 = np.sqrt(np.mean((np.c_[np.ones_like(Yt), Yt] @ c2 - Lt) ** 2))
ypx2L = lambda y: c2[0] + c2[1] * y
log(f"[Fig2] spine x={spine_x}, y={spine_y}; {len(tv)} ticks, log10(TTS) = {c2[0]:.5f} + {c2[1]:.6f}*y_px "
    f"({-1/c2[1]:.2f} px/decade, rms {rms2:.2e} decades)")
xt2 = centers(g[spine_y + 6, spine_x + 3:]); xt2 = [x + spine_x + 3 for x in xt2]
assert len(xt2) == 11, xt2
log(f"[Fig2] x ticks (px): {np.round(xt2,2).tolist()}")

mask = a.sum(2) < 150
mask[:, : spine_x + 2] = False; mask[spine_y - 1:, :] = False
lab, n = ndimage.label(mask)
marks = {}
for i, sl in enumerate(ndimage.find_objects(lab)):
    sz = int((lab[sl] == i + 1).sum()); h = sl[0].stop - sl[0].start; wd = sl[1].stop - sl[1].start
    if not (8 <= sz <= 120 and h <= 9 and wd <= 13): continue
    cy, cx = ndimage.center_of_mass(mask, lab, i + 1)
    j = int(np.argmin([abs(cx - t) for t in xt2])); N = NS[j]
    meth = "QE-MTS" if cx < xt2[j] else "MTS"
    # vertical centre from the full bar extent (robust to partial occlusion by dashed fit lines)
    ycen = (sl[0].start + sl[0].stop - 1) / 2
    marks[(N, meth)] = dict(x_px=cx, y_px=ycen, npx=sz, h=h, w=wd, partial=sz < 45)
missing = [(N, m) for N in NS for m in ("QE-MTS", "MTS") if (N, m) not in marks]
assert not missing, missing

with open(os.path.join(OUT, "fig2_q50_digitized.csv"), "w") as f:
    f.write("# Fig.2 of QMI 8:92 (raster 856x599). Black markers = median-of-medians Q0.50(N,m), in objective evaluations.\n")
    f.write("# y calibrated from 17 axis ticks; marker centre = middle of the bar's pixel extent. 1 px = %.5f decades.\n" % (-c2[1]))
    f.write("# partial=1: bar partly hidden under a dashed fit line (centre from visible extent; check by eye).\n")
    f.write("N,method,Q50_evals,log10_Q50,x_px,y_px,npix,partial\n")
    for N in NS:
        for m in ("QE-MTS", "MTS"):
            r = marks[(N, m)]; L = ypx2L(r["y_px"])
            f.write(f"{N},{m},{10**L:.6g},{L:.5f},{r['x_px']:.2f},{r['y_px']:.2f},{r['npx']},{int(r['partial'])}\n")
log("[Fig2] N  log10Q50(QE)  log10Q50(MTS)  D=QE-MTS   (Fig5 median)")
for N in NS:
    lq = ypx2L(marks[(N, 'QE-MTS')]['y_px']); lm = ypx2L(marks[(N, 'MTS')]['y_px'])
    log(f"   {N}  {lq:.4f}  {lm:.4f}  {lq-lm:+.4f}   ({med[N]:+.4f})"
        + ("  [partial marker]" if marks[(N,'MTS')]['partial'] or marks[(N,'QE-MTS')]['partial'] else ""))

fig, ax = plt.subplots(figsize=(10, 7)); ax.imshow(img)
for (N, m), r in marks.items():
    ax.plot(r["x_px"], r["y_px"], "+", color="k" if m == "MTS" else "m", ms=16, mew=1.2)
for y, v in tv: ax.axhline(y, color="b", lw=0.3)
ax.set_title("Fig.2 check: + = extracted Q0.50 markers (magenta QE-MTS, black MTS); blue = calibrated ticks")
ax.axis("off"); fig.savefig(os.path.join(OUT, "fig2_overlay.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

with open(os.path.join(OUT, "extraction_log.txt"), "w") as f: f.write("\n".join(LOG) + "\n")
