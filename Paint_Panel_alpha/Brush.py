import taichi as ti

@ti.data_oriented
class Brush:
    def __init__(self,res) -> None:
        self.CAN_DRAW = ti.field(ti.i32,shape=())
        self.CAN_DRAW[None] = 1  # can_draw初值为1
        self.SHARP = ti.field(ti.i32,shape=())  # sharp初值为0
        self.res_x=res[0]
        self.res_y=res[1]
        self.posx = ti.field(ti.f32,shape=())
        self.posy = ti.field(ti.f32,shape=())
        self.Zeros = ti.field(ti.f32,res)
        self.Zeros.fill(0)
        self.S=self.Zeros  # S初值为0
        self.pensize = ti.field(ti.f32,shape=())
        self.pensize_init=ti.field(ti.f32,shape=())
        self.pensize_init[None]=10.0
        self.sharp_time = ti.field(ti.i32,shape=())  # 笔锋延续次数参数

        self.sharp_limit = 40  # 笔锋最大延续次数（可调参数）

    # 定义获取gui绘画参数状态值
    @ti.pyfunc
    def get_events(self,CAN_DRAW,SHARP):
        self.CAN_DRAW[None] = CAN_DRAW
        self.SHARP[None] = SHARP

    # 定义获取鼠标位置
    @ti.pyfunc
    def get_pos(self,pos):
        self.posx[None],self.posy[None] = int(pos[0]*self.res_x),int(pos[1]*self.res_y)

    # 定义普通笔刷
    @ti.func
    def general_pen(self,pos_x,pos_y):
        self.pensize[None] = self.pensize_init[None]  # 起始笔刷粗细
        # for i,j in ti.ndrange((pos_x-self.pensize[None],pos_x+self.pensize[None]),(pos_y-self.pensize[None],pos_y+self.pensize[None])):
        for i,j in ti.ndrange((0,self.res_x),(0,self.res_y)):
            r = ((i-pos_x)**2+(j-pos_y)**2)**0.5
            if r <= self.pensize[None]:
                self.S[i,j] = 0.5*(1+1/(r+1))

    # 定义笔锋
    @ti.func
    def edge_pen(self,pos_x,pos_y):
        # bimo = self.Zeros
        self.pensize[None] = self.pensize_init[None]  # 起始笔刷粗细
        self.pensize[None] = self.pensize[None] / (( self.sharp_time[None] / self.sharp_limit )+1)
        # for i,j in ti.ndrange((pos_x-self.pensize[None],pos_x+self.pensize[None]),(pos_y-self.pensize[None],pos_y+self.pensize[None])):
        for i,j in ti.ndrange((0,self.res_x),(0,self.res_y)):
            r = ((i-pos_x)**2+(j-pos_y)**2)**0.5
            if self.sharp_time[None] <= self.sharp_limit:
                if r <= self.pensize[None]:
                    self.S[i,j] = 0.5*(1+1/(r+1))
                else:
                    self.S[i,j] = 0

    # 定义绘画S值输出
    @ti.kernel
    def general_S(self):
        if self.CAN_DRAW[None] == 0:
            if self.SHARP[None] == 0:
                self.general_pen(self.posx[None],self.posy[None])
                self.sharp_time[None] = 0
            elif self.SHARP[None] == 1:
                self.sharp_time[None] += 1
                self.edge_pen(self.posx[None],self.posy[None])  