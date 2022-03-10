import numpy as np
import taichi as ti
from inkcanvas import inkCanvas
RES = 512
ti.init(arch = ti.gpu)
gui = ti.GUI(name = "ink", res = RES)
canv = inkCanvas(RES)

while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.LMB:
            pos = gui.get_cursor_pos()
            canv.add(100, pos[0]*RES, pos[1]*RES)
    for i in range(10):
        canv.simluate()
    canv.render()
    gui.set_image(canv.pixels)
    #print(canv.pixels)
    print(canv.particle_cnt)
    gui.show()