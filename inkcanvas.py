import numpy as np
import taichi as ti
particle_num = 10000
step_len = 0.5
@ti.data_oriented
class inkCanvas:
    def __init__(self, res):
        self.res = res
        self.pixels = ti.field(ti.f32, shape = (res, res))
        self.particle_cnt = ti.field(ti.i32, shape = ())
        self.particle_cnt[None] = 0
        self.particles = ti.Vector.field(2, shape = particle_num, dtype = ti.f32)

    @ti.kernel
    def add(self, _num:ti.i32, _x:ti.f32, _y:ti.f32):
        if self.particle_cnt[None] + _num <= particle_num:
            for i in range(self.particle_cnt[None], self.particle_cnt[None] + _num):
                self.particles[i] = (_x, _y)
            self.particle_cnt[None] += _num
    
    @ti.kernel
    def simluate(self):
        for p in range(self.particle_cnt[None]):
            theta = 2 * 3.1415926 * ti.random(dtype=ti.f32)
            diff = (ti.cos(theta), ti.sin(theta))
            self.particles[p][0] += diff[0] * step_len
            self.particles[p][1] += diff[1] * step_len
    
    @ti.func
    def density(self, va, vb):
        diff = (va - vb).norm()
        diff /= self.res
        return 0.1*ti.exp(-10000*(diff))

    @ti.kernel
    def render(self):
        for i, j in ti.ndrange(self.res, self.res):
            self.pixels[i, j] = 1
        for i, j, k in ti.ndrange(self.res, self.res, self.particle_cnt[None]):
            self.pixels[i, j] -= self.density((i, j), self.particles[k]) * 0.5
