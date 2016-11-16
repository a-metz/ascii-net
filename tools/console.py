def draw(iterable2d):
    for row in iterable2d:
        out = ''
        for col in row:
            if col >= 0.5:
                out += '#'
            else:
                out += '.'
        print(out)
