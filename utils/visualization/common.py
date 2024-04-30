import math


def draw_arrowed_line(draw, ptA, ptB, color=(0, 255, 0), width=2):
    """Draw line from ptA to ptB with arrowhead at ptB"""
    # Draw the line without arrows
    draw.line((ptA, ptB), width=width, fill=color)

    # Draw arrow from ptA to ptB (in general in the text direction)
    draw_triangle_on_the_edge(draw, ptA, ptB, color=color)

def draw_triangle_on_the_edge(draw, ptA, ptB, color=(0, 255, 0)):
    # Now work out the arrowhead
    # = it will be a triangle with one vertex at ptB
    # - it will start at 95% of the length of the line
    # - it will extend 8 pixels either side of the line
    x0, y0 = ptA
    x1, y1 = ptB
    # Now we can work out the x,y coordinates of the bottom of the arrowhead triangle
    xb = 0.95 * (x1 - x0) + x0
    yb = 0.95 * (y1 - y0) + y0

    # Work out the other two vertices of the triangle
    # Check if line is vertical
    if x0 == x1:
        vtx0 = (xb - 5, yb)
        vtx1 = (xb + 5, yb)
    # Check if line is horizontal
    elif y0 == y1:
        vtx0 = (xb, yb + 5)
        vtx1 = (xb, yb - 5)
    else:
        alpha = math.atan2(y1 - y0, x1 - x0) - 90 * math.pi / 180
        a = 8 * math.cos(alpha)
        b = 8 * math.sin(alpha)
        vtx0 = (xb + a, yb + b)
        vtx1 = (xb - a, yb - b)

    # Now draw the arrowhead triangle
    draw.polygon([vtx0, vtx1, ptB], fill=color)
