import colorful

colors = {'black': (0, 0, 0)}
for i in range(0, 256):
    colors[str(i)] = (i, 255, i)
    colors[str(i + 256)] = (255, 255-i, 255-i)
colorful.use_palette(colors)
colorful.update_palette(colors)

print (getattr(colorful, f'black_on_{i}')(char), end='')
