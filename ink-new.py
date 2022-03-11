import numpy as np
import taichi as ti
import time
from inkcanvas import inkCanvas
RES = 300 #屏幕范围
ti.init(arch = ti.cpu)

gui = ti.GUI(name = "ink", res = RES)
canv = inkCanvas(RES)
ss = 0
flag = False
print("generating the fibre...")
canv.fibre_gen(0.8, 0.3)
print("generted!")

while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.LMB:
            flag = not flag
        if e.key == 'c':
            canv.particle_cnt[None] = 0

    if flag:
        pos = gui.get_cursor_pos()
        canv.add(10, pos[0]*RES, pos[1]*RES)
        if ss == 0:
            ss = 1
    if ss != 0:
        ss += 1
    for i in range(5):#sub frame
        canv.simluate()
    canv.render()
    gui.set_image(canv.pixels)
    #print(canv.pixels)
    print(canv.particle_cnt)
    filename = f'./result1/frame_{ss:04d}.png'
    gui.show()
