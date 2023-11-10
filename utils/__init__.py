def get_bounding_box_coords(center, height, width):
    left_x = center - width // 2
    top_y = center - height // 2

    tl = (left_x, top_y)
    tr = (left_x + width, top_y)
    bl = (left_x, top_y + height)
    br = (left_x + width, top_y + height)

    return [[tl, tr], [bl, br]]