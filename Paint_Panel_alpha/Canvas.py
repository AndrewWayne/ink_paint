import taichi as ti
import numpy as np
S_and_K_percent=3
color_filter=0.8
dh = 1 # 空间步长
dt = 0.05 # 时间步长
dir_x = np.array([-1, 1, 0, 0, -1, 1, -1, 1])
dir_y = np.array([0, 0, -1, 1, 1, -1, -1, 1])
@ti.data_oriented
class Canvas:
    def __init__(self, fibre):
        D_ratio=61  # 方便调参用

        self.edged_res_x = len(fibre)
        self.edged_res_y = len(fibre[0])
        self.res_x=self.edged_res_x-2
        self.res_y=self.edged_res_y-2
        self.res = (self.res_x,self.res_y)  # 分辨率
        self.edged_res = (self.edged_res_x,self.edged_res_y)  # 镶边处理
        self.grids = ti.field(ti.f32, self.edged_res)  # 存储墨水量的格点
        self.grids.fill(0)  # grids初值为0
        self.nabla_D = ti.Vector.field(2, ti.f32, self.res)  # 存储扩散系数散度 (这里要转成list来存元组)
        self.sources = ti.field(ti.f32, self.res)  # 存储此刻笔刷效果
        self.sources.fill(0)  # sources初值为0
        self.ratios = ti.field(ti.f32, self.edged_res)  # 按墨水量分配的比例
        self.pixels= ti.Vector.field(3,ti.f32, self.res)  # 像素格
        self.prepixels = ti.field(ti.f32, self.res) # 处理边缘浓度
        self.D = ti.field(ti.f32, self.edged_res)
        self.D.from_numpy(fibre * D_ratio)
        self.dir_x = ti.field(ti.i32, shape=8)
        self.dir_x.from_numpy(dir_x*3)
        self.dir_y = ti.field(ti.i32, shape=8)
        self.dir_y.from_numpy(dir_y*3)



    @ti.func
    def absorption(self, water):  # 调整全平面基准散度 K 使得水量较低的时候更加顺滑，且让纸不会把水量吸到过大的负值，从而影响吸水效果。
        water_flag = -0.1
        max_absorption = 0.05
        resu = 0.0
        if water >= water_flag:
            resu = 0.15
        return resu

    @ti.func
    def tint(self, ratio, d):  # 根据墨水量0-1的比例调配出合理灰度的颜色
        ans = 0.0
        if ratio >= 0.3:
            ans = ti.exp(18*(ti.min(1, ratio)-1))*d
        return ans

    @ti.func
    def nabla(self, grids,x,y):
        partial_x = (grids[x+1, y] - grids[x-1, y]) / (2 * dh)
        partial_y = (grids[x, y+1] - grids[x, y-1]) / (2 * dh)
        return ti.Vector([partial_x, partial_y])
    
    @ti.func
    def inner_product(self, vector1, vector2):
        return (vector1[0]*vector2[0] + vector1[1]*vector2[1])

    @ti.func
    def nabla_2(self, grids, x,y):
        partial_x_2 = (grids[x+1, y] + grids[x-1, y] - 2*grids[x, y])/(dh*dh)
        partial_y_2 = (grids[x, y+1] + grids[x, y-1] - 2*grids[x, y])/(dh*dh)
        return partial_x_2 + partial_y_2
    
    @ti.func
    def get_S(self,S):
        self.sources = S
    #周围8+12格加权平均，得到最终中心墨色
    @ti.func
    def smooth_func(self, x, y):
        self.pixels[x, y][0] = self.prepixels[x, y]
        if self.pixels[x, y][0] >= 0.3:
            for idx in range(8):
                ux = x + self.dir_x[idx]
                uy = y + self.dir_y[idx]
                self.pixels[x, y][0] += self.prepixels[ux, uy] * 0.7
            self.pixels[x, y][0] /= 6.6
            ti.atomic_max(self.pixels[x, y][0], 0.5)
            ti.atomic_sub(self.pixels[x, y][0], 0.5)
            self.pixels[x, y][0] *= 2
        ti.atomic_max(self.pixels[x, y][0], 0)
        self.pixels[x, y][0] -= 1
        self.pixels[x, y][0] *= -1
        _pixel_num=255.0*self.pixels[x, y][0]
        self.pixels[x, y]=[_pixel_num,_pixel_num,_pixel_num]
        #self.pixels[x, y] = self.tint(self.pixels[x, y], 1.1)

    # 计算nabla D
    @ti.kernel
    def Init(self):
        for x, y in ti.ndrange(self.res_x, self.res_y):
            self.nabla_D[x, y]= self.nabla(self.D, x+1, y+1)

    # 更新墨水分布   
    @ti.func
    def Update(self):
        for x, y in ti.ndrange(self.res_x-1, self.res_y-1):
        # 这里写的很有问题，下标没有统一看起来很丑。girds是定义在edged-res上的，nablaD由于中心差分的原因则是定义在res上的（改进一下吧）
        # 这个地方据说要用稀疏矩阵来优化，还得学一学
            self.ratios[x+2, y+2] = self.grids[x+2, y+2] + dt * ( self.D[x+2, y+2]*self.nabla_2(self.grids, x+2, y+2) + self.inner_product(self.nabla(self.grids, x+2, y+2), self.nabla_D[x+1, y+1]) + S_and_K_percent * (self.sources[x+1, y+1] - self.absorption(self.grids[x+2, y+2]))) + self.sources[x+1, y+1]
        #ratios[x+2, y+2] = grids[x+2, y+2] + dt * (D[x+2, y+2]*inner_product(nabla(grids, x+2, y+2), ti.Vector([1, 1]))  + S_and_K_percent * (sources[x+1, y+1] - absorption(grids[x+2, y+2])))
            self.sources[x+1, y+1] = 0

    # 将ratios归到0-1之间，防止异常错误 element_wise
        for i, j in ti.ndrange(self.res_x, self.res_y):
            ti.atomic_max(self.ratios[i+1, j+1], 0.0)
            ti.atomic_min(self.ratios[i+1, j+1], 0.9999)

        for i, j in ti.ndrange(self.res_x, self.res_y):
            self.grids[i+1, j+1] = self.ratios[i+1, j+1]  # 把值再还给grids

    # sources=np.zeros(res) # 赋一次值之后置0看看

    # 用1-1/x 来叠加，使得墨色只能变深
        for x in range(self.res_x+2):
            self.grids[x, 0], self.grids[x, self.res_y+1] = 0.0, 0.0

        for y in range(self.res_y+2):
            self.grids[0, y], self.grids[self.res_x+1, y] = 0.0, 0.0
    
        for x, y in ti.ndrange(self.res_x, self.res_y):
            self.prepixels[x, y] += self.tint(self.ratios[x+1, y+1], 1)/3
            ti.atomic_min(self.prepixels[x, y], 1.0)

        for x, y in ti.ndrange(self.res_x-2, self.res_y-2):
            px, py = x+1, y+1
            self.smooth_func(px, py)


        
        
    