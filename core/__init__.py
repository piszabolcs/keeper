# Keeper - image processing core. No Qt imports below this package.

from PIL import Image

import config

# Panoramas above Pillow's ~179 MP "decompression bomb" limit are legitimate
# photos here; the limit is raised deliberately, not removed (C-12).
Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS
