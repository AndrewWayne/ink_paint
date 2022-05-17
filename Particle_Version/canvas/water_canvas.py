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

    @ti.func
    def step_length(self, _x, _y):
        newx, newy = int(_x), int(_y)
        return step_len * (1 - self.fibre_map[newx, newy])

    @ti.kernel
    def fibre_gen(self, d:ti.f64, f:ti.f64):#We recommend d = 0.8, f = 0.3
        # init
        for i, j in ti.ndrange(self.res, self.res):
            self.fibre_map[i, j] = 0
        
    @ti.kernel
    def simluate(self):
        for p in range(self.particle_cnt[None]):
            theta = int(2*ti.random(dtype=ti.f32))
            pos = 2*(int(ti.random(ti.f32) * 2) - 0.5)
            if theta == 0:
                self.particles[p][0] += self.step_length(self.particles[p][0], self.particles[p][1]) * pos
            else:
                self.particles[p][1] += self.step_length(self.particles[p][0], self.particles[p][1]) * pos

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
            if self.pixels[i, j] > 0.5:#filter 使边缘光滑
                self.pixels[i, j] = 1
