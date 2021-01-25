import math
from numpy import random
from PIL import Image, ImageDraw, ImagePath
from pathlib import Path


def clip(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def coordinate():
    side = random.randint(3, 6)
    # 坐标加1把单位圆从原点移动到第一象限
    xy = [(math.cos(th) + 1, math.sin(th) + 1)
          for th in [s * (2 * math.pi) / side for s in range(side)]]

    loc, scale = 75, 10
    x_scale, y_scale = random.normal(loc, scale), random.normal(loc, scale)
    x_shift, y_shift = random.uniform(0, 800), random.uniform(0, 800)
    xy = list(map(lambda coords: (coords[0] * x_scale + x_shift,
                                  coords[1] * y_scale + y_shift),
                  xy))
    if random.randn() > 0.25:
        return ImagePath.Path(xy).getbbox()
    return xy


def retouch(coords, color):
    m_coords, m_color = [], []
    if random.randn() > 0:
        if isinstance(coords, tuple):
            for c in coords:
                m_coords.append(c + random.normal(0, 15))
        else:
            for c in coords:
                m_coords.append((c[0] + random.normal(0, 15),
                                 c[1] + random.normal(0, 15)))

        m_color = color
    else:
        m_coords = coords
        for c in color:
            m_color.append(int(clip(c + random.normal(0, 50), 0, 255)))
    return (tuple(m_coords) if isinstance(coords, tuple) else m_coords,
            tuple(m_color))


def draw(draw, coords, color):
    if isinstance(coords, tuple):
        if random.randn() > 0:
            draw.rectangle(coords, fill=color)
        else:
            draw.ellipse(coords, fill=color)
    else:
        draw.polygon(coords, fill=color)


def simulate(directory):
    count = 25
    size_fine = 1024
    ref_fine = Image.new("RGB", (size_fine, size_fine), (255, 255, 255))
    pred_fine = Image.new("RGB", (size_fine, size_fine), (255, 255, 255))
    draw_ref, draw_pred = ImageDraw.Draw(ref_fine), ImageDraw.Draw(pred_fine)
    for _ in range(count):
        color_ref = (random.randint(0, 255),
                     random.randint(0, 255),
                     random.randint(0, 255))
        xy_ref = coordinate()
        xy_pred, color_pred = retouch(xy_ref, color_ref)
        draw(draw_ref, xy_ref, color_ref)
        draw(draw_pred, xy_pred, color_pred)
    size_coarse = size_fine // 16
    ref_coarse = ref_fine.resize((size_coarse, size_coarse), resample=Image.LANCZOS)
    pred_coarse = pred_fine.resize((size_coarse, size_coarse), resample=Image.LANCZOS)
    ref_fine.save(str(directory / 'ref_fine.png'))
    ref_coarse.save(str(directory / 'ref_coarse.png'), quality=100)
    pred_fine.save(str(directory / 'pred_fine.png'))
    pred_coarse.save(str(directory / 'pred_coarse.png'), quality=100)


if __name__ == '__main__':
    directory = Path('/Users/TheOneGIS/Desktop/SIM')
    if not directory.exists():
        directory.mkdir()
    for i in range(100):
        name = '{:03d}'.format(i)
        subdir = directory / name
        subdir.mkdir()
        simulate(subdir)



