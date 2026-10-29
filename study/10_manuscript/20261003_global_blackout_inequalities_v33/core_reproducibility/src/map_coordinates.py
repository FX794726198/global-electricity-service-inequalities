"""Geographic ticks at the true intersections with a projected map frame.

Use after setting the final projected extent. Bottom labels are meridians;
left labels are parallels. Top/right retain matching geographic ticks without
duplicate labels. Never label projected metres as longitude or latitude.
"""
import numpy as np
from pyproj import Transformer


def _values(a, b, count=4):
    span = b - a
    candidates = np.array([.1, .2, .25, .5, 1, 2, 2.5, 5, 10, 15, 20, 30, 45])
    step = candidates[np.argmin(np.abs(candidates - span / count))]
    return np.arange(np.ceil(a / step) * step, b + step * .01, step)


def geographic_frame(ax, crs, longitudes=None, latitudes=None, fontsize=9,
                     grid=True, label_bottom=True, label_left=True):
    """Add projected graticules and geographical border ticks; return audit data."""
    forward = Transformer.from_crs(4326, crs, always_xy=True)
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx = np.r_[np.linspace(x0, x1, 200), np.repeat(x1, 200), np.linspace(x1, x0, 200), np.repeat(x0, 200)]
    by = np.r_[np.repeat(y0, 200), np.linspace(y0, y1, 200), np.repeat(y1, 200), np.linspace(y1, y0, 200)]
    lons, lats = inverse.transform(bx, by)
    lon_min, lon_max = np.nanmin(lons), np.nanmax(lons)
    lat_min, lat_max = np.nanmin(lats), np.nanmax(lats)
    if longitudes is None:
        longitudes = _values(lon_min, lon_max)
    if latitudes is None:
        latitudes = _values(lat_min, lat_max)
    ticks = {'bottom': [], 'top': [], 'left': [], 'right': []}

    def crossings(v, w, edge, low, high):
        valid = np.isfinite(v[:-1]) & np.isfinite(v[1:]) & np.isfinite(w[:-1]) & np.isfinite(w[1:])
        ids = np.flatnonzero(valid & ((v[:-1] - edge) * (v[1:] - edge) <= 0) & (np.abs(np.diff(v)) > 1e-10))
        out = []
        for i in ids:
            z = w[i] + (edge - v[i]) * (w[i + 1] - w[i]) / (v[i + 1] - v[i])
            # Avoid corner collisions and degenerate edge intersections.
            if low + .035 * (high - low) < z < high - .035 * (high - low):
                if not out or abs(z - out[-1]) > 1e-5 * (high - low):
                    out.append(float(z))
        return out

    for degree in longitudes:
        yy = np.linspace(max(-89, lat_min - 5), min(89, lat_max + 5), 3001)
        xx, yy = forward.transform(np.full_like(yy, degree), yy)
        if grid:
            ax.plot(xx, yy, color='#737D82', lw=.35, alpha=.27, ls=(0, (3, 3)), zorder=5, clip_on=True)
        for side, edge in [('bottom', y0), ('top', y1)]:
            ticks[side].extend((v, float(degree)) for v in crossings(yy, xx, edge, x0, x1))
    for degree in latitudes:
        xx = np.linspace(lon_min - 5, lon_max + 5, 3001)
        xx, yy = forward.transform(xx, np.full_like(xx, degree))
        if grid:
            ax.plot(xx, yy, color='#737D82', lw=.35, alpha=.27, ls=(0, (3, 3)), zorder=5, clip_on=True)
        for side, edge in [('left', x0), ('right', x1)]:
            ticks[side].extend((v, float(degree)) for v in crossings(xx, yy, edge, y0, y1))

    def fmt(degree, longitude):
        if abs(degree) < 1e-10:
            return '0°'
        suffix = ('E' if degree > 0 else 'W') if longitude else ('N' if degree > 0 else 'S')
        return f'{abs(degree):g}°{suffix}'

    for side in ticks:
        ticks[side].sort()
    ax.set_axis_on()
    ax.set_xticks([p for p, _ in ticks['bottom']])
    ax.set_xticklabels([fmt(d, True) for _, d in ticks['bottom']], fontsize=fontsize)
    ax.set_yticks([p for p, _ in ticks['left']])
    ax.set_yticklabels([fmt(d, False) for _, d in ticks['left']], fontsize=fontsize)
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=label_bottom, direction='out', length=2.4, pad=2, width=.65)
    ax.tick_params(axis='y', left=True, right=False, labelleft=label_left, direction='out', length=2.4, pad=2, width=.65)
    # Top/right intersections differ from bottom/left under the projection.
    for v, d in ticks['top']:
        ax.plot([v, v], [y1, y1 + .011 * (y1 - y0)], color='#5E6365', lw=.65, clip_on=False, zorder=12)
    for v, d in ticks['right']:
        ax.plot([x1, x1 + .009 * (x1 - x0)], [v, v], color='#5E6365', lw=.65, clip_on=False, zorder=12)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#65696A')
        spine.set_linewidth(.65)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    errors = []
    for side, pairs in ticks.items():
        for pos, degree in pairs:
            lon, lat = inverse.transform(pos if side in ['bottom', 'top'] else (x0 if side == 'left' else x1), (y0 if side == 'bottom' else y1) if side in ['bottom', 'top'] else pos)
            errors.append(abs((lon if side in ['bottom', 'top'] else lat) - degree))
    return {'crs': str(crs), 'longitudes': list(map(float, longitudes)), 'latitudes': list(map(float, latitudes)), 'projected_frame_intersections': {k: [{'position': p, 'degrees': d} for p, d in v] for k, v in ticks.items()}, 'max_roundtrip_error_degrees': max(errors, default=0), 'labelled_borders': ['bottom longitude', 'left latitude'], 'top_right': 'true geographic intersection ticks, labels omitted to avoid duplication'}
