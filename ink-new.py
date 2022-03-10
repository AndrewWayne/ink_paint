import numpy as np
import taichi as ti
from inkcanvas import inkCanvas
RES = 300
ti.init(arch = ti.gpu)
gui = ti.GUI(name = "ink", res = RES)
canv = inkCanvas(RES)
ss = 0
flag = False
while gui.running:

    if gui.get_event(ti.GUI.LMB):
        flag = not flag

    if flag:
        pos = gui.get_cursor_pos()
        canv.add(50, pos[0]*RES, pos[1]*RES)
        if ss == 0:
            ss = 1
    if ss != 0:
        ss += 1
    for i in range(10):
        canv.simluate()
    canv.render()
    gui.set_image(canv.pixels)
    #print(canv.pixels)
    print(canv.particle_cnt)
    gui.show(f'./result2/ink{ss:05d}.png')