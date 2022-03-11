import numpy as np
import taichi as ti
particle_num = 20000
step_len = 0.5 #扩散速率
@ti.data_oriented
class inkCanvas:
    def __init__(self, res):
        self.res = res
        self.pixels = ti.field(ti.f32, shape = (res, res))
        self.particle_cnt = ti.field(ti.i32, shape = ())
        self.particle_cnt[None] = 0
        self.particles = ti.Vector.field(2, shape = particle_num, dtype = ti.f32)
        self.fibre_map = ti.field(ti.f32, shape = (res, res))
        self.fibre_knot = ti.Vector.field(3, shape = int(res/10) ** 2, dtype = ti.i32)
        self.knot_num = ti.field(ti.i32, shape = ())

    @ti.kernel
    def add(self, _num:ti.i32, _x:ti.f32, _y:ti.f32):
        if self.particle_cnt[None] + _num <= particle_num:
            for i in range(self.particle_cnt[None], self.particle_cnt[None] + _num):
                self.particles[i] = (_x, _y)
            self.particle_cnt[None] += _num

    @ti.kernel
    def fibre_gen(self, d:ti.f64, f:ti.f64):#We recommend d = 0.8, f = 0.3
        # init
        for i, j in ti.ndrange(self.res, self.res):
            self.fibre_map[i, j] = 0
        # generate fibre knot
        self.knot_num[None] = 0
        for i, j in ti.ndrange(int(self.res / 10), int(self.res / 10)):
            _x = int(ti.random() * 10 + i * 10)
            _y = int(ti.random() * 10 + j * 10)
            self.fibre_knot[self.knot_num[None]] = (_x, _y, 0)
            self.knot_num[None] += 1
            self.fibre_map[_x, _y] = d
        #generate fibre
        for i in range(self.knot_num[None]):
            pos0 = self.fibre_knot[i]
            if pos0[2] >= 2:
                continue
            for cnt in range(2 - pos0[2]):
                k = int(ti.randn() * self.knot_num[None])
                while k == i or self.fibre_knot[k][2] >= 2:
                    k = int(ti.randn() * self.knot_num[None])
                pos1 = self.fibre_knot[k]
                self.fibre_knot[k][2] += 1
                self.fibre_knot[i][2] += 1
                lin1 = (pos1[1] - pos0[1]) / (pos1[0] - pos0[0])
                for _x, _y in ti.ndrange(self.res, self.res):
                    lin3 = (_y - pos0[1]) / (_x - pos0[0])
                    lin4 = (_y - pos1[1]) / (_x - pos1[0])
                    if (ti.abs(lin3 - lin1) <= 1e-3 or ti.abs(lin4 - lin1) <= 1e-3) and _x >= ti.min(pos0[0], pos1[0]) and _x <= ti.max(pos0[0], pos1[0]):
                        self.fibre_map[_x, _y] = ti.max(self.fibre_map[_x, _y], f)

    @ti.kernel
    def simluate(self):
        for p in range(self.particle_cnt[None]):
            theta = int(2*ti.random(dtype=ti.f32))
            pos = 2*(int(ti.random(ti.f32) * 2) - 0.5)
            if theta == 0:
                self.particles[p][0] += step_len * pos
            else:
                self.particles[p][1] += step_len * pos

    @ti.func
    def density(self, va, vb):
        diff = (va - vb).norm()
        diff /= self.res
        return 0.1*ti.exp(-10000*(diff*diff)) #距离插值函数

    @ti.kernel
    def render(self):
        for i, j in ti.ndrange(self.res, self.res):
            self.pixels[i, j] = 1
        for i, j, k in ti.ndrange(self.res, self.res, self.particle_cnt[None]):
            self.pixels[i, j] -= self.density((i, j), self.particles[k])
        for i, j in ti.ndrange(self.res, self.res):
            if self.pixels[i, j] > 0.7:#filter 使边缘光滑
                self.pixels[i, j] = 1
