import numpy as np
import taichi as ti
particle_num = 20000
dt = 0.1
dh = 1
Radium = 10
_offset = 500 #偏置常数
@ti.data_oriented
class inkCanvas:
    def __init__(self, res):
        self.res = res
        self.pixels = ti.field(ti.f32, shape = (res, res))
        self.inkdense = ti.Vector.field(2, dtype = ti.f32, shape = (res, res))
        self.tmpdense = ti.Vector.field(2, dtype = ti.f32, shape = (res, res))
        self.fibre_map = ti.field(ti.f32, shape = (res, res))
        self.fibre_knot = ti.Vector.field(3, shape = int(res/10) ** 2, dtype = ti.i32)
        self.knot_num = ti.field(ti.i32, shape = ())

    @ti.kernel
    def add(self, _dense:ti.i32, _x:ti.i32, _y:ti.i32):
        for i, j in ti.ndrange(Radium, Radium):
            if i * i + j * j <= Radium:
                p1, p2, p3, p4 = (_x - i, _y -j), (_x + i, _y - j), (_x - i, _y + j), (_x + i, _y + j)
                i1 = self.inkdense[p1[0], p1[1]]
                i2 = self.inkdense[p2[0], p2[1]]
                i3 = self.inkdense[p3[0], p3[1]]
                i4 = self.inkdense[p4[0], p4[1]]
                self.inkdense[p1[0], p1[1]][0] = i1[0] * self.density(i1[0], i1[1]) + _dense
                self.inkdense[p1[0], p1[1]][1] = 0
                self.inkdense[p2[0], p2[1]][0] = i2[0] * self.density(i2[0], i2[1]) + _dense
                self.inkdense[p2[0], p2[1]][1] = 0
                self.inkdense[p3[0], p3[1]][0] = i3[0] * self.density(i3[0], i3[1]) + _dense
                self.inkdense[p3[0], p3[1]][1] = 0
                self.inkdense[p4[0], p4[1]][0] = i4[0] * self.density(i4[0], i4[1]) + _dense
                self.inkdense[p4[0], p4[1]][1] = 0

    @ti.kernel
    def fibre_gen(self, d:ti.f64, f:ti.f64):#We recommend d = 0.8, f = 0.3
        for i, j in ti.ndrange(self.res, self.res):
            self.inkdense[i, j] = (0, 0)
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
                for _x in range(self.res):
                    for _y in range(self.res):
                        lin3 = (_y - pos0[1]) / (_x - pos0[0])
                        lin4 = (_y - pos1[1]) / (_x - pos1[0])
                        if (ti.abs(lin3 - lin1) <= 1e-3 or ti.abs(lin4 - lin1) <= 1e-3) and _x >= ti.min(pos0[0], pos1[0]) and _x <= ti.max(pos0[0], pos1[0]):
                            self.fibre_map[_x, _y] = ti.max(self.fibre_map[_x, _y], f)

    @ti.kernel
    def simluate(self):
        for i, j in ti.ndrange(self.res, self.res):
            self.tmpdense[i, j] = self.inkdense[i, j]
            self.tmpdense[i, j][1] += 1
            self.tmpdense[i, j][0] -= 4 * dt * self.SpreadCoe(i, j) * self.inkdense[i, j][0]
        for i, j in ti.ndrange(self.res - 1, self.res - 1):
            self.tmpdense[i, j][0] += dt * self.SpreadCoe(i+1, j) * self.inkdense[i+1, j][0] + dt * self.SpreadCoe(i, j+1) * self.inkdense[i, j+1][0]
            self.tmpdense[i+1, j][0] += dt * self.SpreadCoe(i, j) * self.inkdense[i, j][0] 
            self.tmpdense[i, j+1][0] += dt * self.SpreadCoe(i, j) * self.inkdense[i, j][0]
        for i in range(self.res - 1):
            self.tmpdense[i, self.res - 1][0] += dt * self.SpreadCoe(i+1, self.res - 1) * self.inkdense[i+1, self.res - 1][0]
            self.tmpdense[self.res - 1, i][0] += dt * self.SpreadCoe(self.res - 1, i+1) * self.inkdense[self.res - 1, i+1][0]
        for i, j in ti.ndrange(self.res, self.res):
            self.inkdense[i, j] = self.tmpdense[i, j]

    @ti.func
    def density(self, _Q, _t):
        dens = ti.cos(3.14 * _t / (2 * (_Q + _offset))) 
        if _t >= _Q + _offset:
            dens = 0
        return dens #墨水浓度函数

    @ti.func
    def SpreadCoe(self, _x, _y):
        return self.density(self.inkdense[_x, _y][0], self.inkdense[_x, _y][1]) * (1 - self.fibre_map[_x, _y])

    @ti.kernel
    def render(self):
        for i, j in ti.ndrange(self.res, self.res):
            self.pixels[i, j] = 0 if self.inkdense[i, j][0] / 1000 > 1 else 1 - self.inkdense[i, j][0] / 1000
