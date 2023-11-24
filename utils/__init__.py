def get_bounding_box_coords(center, height, width):
    left_x = center[0] - width // 2
    top_y = center[1] - height // 2

    bl = (left_x, top_y)
    br = (left_x + width, top_y)
    tl = (left_x, top_y + height)
    tr = (left_x + width, top_y + height)

    return [[bl, br], [tl, tr]]