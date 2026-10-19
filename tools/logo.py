# Keeper - derives the committed logo assets from the raw sources.
#
#   .venv\Scripts\python.exe tools\logo.py
#
# The raw sources (multi-MB PNGs) live in the git-ignored branding/ folder;
# the small derivatives in assets/ are committed, so a fresh clone works and
# the README shows its header without running anything.

import pathlib

from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
BRANDING = ROOT / "branding"

SOURCE_WITH_NAME = BRANDING / "LOGO_nobg.png"   # badge with the "Keeper" lettering
SOURCE_MARK = BRANDING / "LOGO_icon.png"        # the aperture mark only

README_WIDTH = 460
README_PADDING = 26
# the app's background colours (gui/theme.py): the logo nearly disappears on
# a light GitHub theme, so the README header gets the dark background baked in
BACKGROUND_TOP = (17, 18, 20)
BACKGROUND_BOTTOM = (23, 24, 27)
CORNER_RADIUS = 22
ICON_SIZES = [16, 20, 24, 32, 48, 64, 128, 256]
MARK_SIZE = 96


def trimmed(image):
    """Drops the transparent margin, so a requested size applies to the mark itself."""
    box = image.getbbox()
    return image.crop(box) if box else image


def scaled(image, width):
    return image.resize((width, max(1, round(image.height * width / image.width))), Image.LANCZOS)


def rounded_background(width, height):
    background = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(background)
    for y in range(height):
        t = y / max(1, height - 1)
        draw.line([(0, y), (width, y)], fill=tuple(
            round(BACKGROUND_TOP[i] + (BACKGROUND_BOTTOM[i] - BACKGROUND_TOP[i]) * t) for i in range(3)
        ))
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle([(0, 0), (width - 1, height - 1)], radius=CORNER_RADIUS, fill=255)
    result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    result.paste(background, (0, 0), mask)
    return result


def readme_header():
    with Image.open(SOURCE_WITH_NAME) as source:
        logo = scaled(trimmed(source.convert("RGBA")), README_WIDTH)
    image = rounded_background(logo.width + 2 * README_PADDING, logo.height + 2 * README_PADDING)
    image.alpha_composite(logo, (README_PADDING, README_PADDING))
    target = ASSETS / "logo.png"
    image.save(target, optimize=True)
    return target


def windows_icon():
    with Image.open(SOURCE_MARK) as source:
        mark = trimmed(source.convert("RGBA"))
    side = max(mark.width, mark.height)   # square, or the .ico would distort
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.alpha_composite(mark, ((side - mark.width) // 2, (side - mark.height) // 2))
    target = ASSETS / "keeper.ico"
    square.save(target, format="ICO", sizes=[(s, s) for s in ICON_SIZES])
    return target


def app_mark():
    with Image.open(SOURCE_MARK) as source:
        mark = scaled(trimmed(source.convert("RGBA")), MARK_SIZE * 2)
    target = ASSETS / "logo_mark.png"
    mark.save(target, optimize=True)
    return target


def main():
    for build in (readme_header, windows_icon, app_mark):
        target = build()
        print("%-24s %8.1f KB" % (target.name, target.stat().st_size / 1024))


if __name__ == "__main__":
    main()
