import taichi as ti
import sys
import os
import Canvas
import Brush
#import cv2
import matplotlib.image as im
import numpy as np
import datetime

@ti.data_oriented
class Panel:
    def __init__(self,fibre) -> None:
        self.fibre=fibre
        res_x=len(fibre)-2
        res_y=len(fibre[0])-2
        self.res_x=res_x
        self.res_y=res_y
        res = (res_x,res_y)
        self.res=res
        self.canvas=Canvas.Canvas(fibre)
        self.brush=Brush.Brush(res)
        self.gui=ti.GUI(name = "ink", res = res,fast_gui=True)
        self.canvas.Init()
        self.click_l = ti.field(ti.i32,shape=())
        self.click_r = ti.field(ti.i32,shape=())
        self.can_draw = ti.field(ti.i32,shape=())
        self.bifeng = ti.field(ti.i32,shape=())
        self.can_draw[None] = 1

        self.pic_order=0

    # 处理交互参数
    def gui_judge(self):
        self.gui.get_event()

        if self.gui.is_pressed('e'):
            self.click_l[None] = 1
        if self.click_l[None] == 1 and self.gui.is_pressed('f'):
            self.click_r[None] = 1
        if self.click_l[None] == 1 and self.gui.is_pressed('r'):
            self.click_l[None] = 0
            self.click_r[None] = 0
            self.brush.pensize_init[None]=10.0
        
        if self.click_l[None] == 1:  # 左键按下开始书写
            self.can_draw[None] = 0
        if self.click_r[None] == 1 and self.bifeng[None] == 0:  # 右键按下开始笔锋
            self.bifeng[None] = 1
        if self.click_l[None] == 0 and self.click_r[None] == 0:  # 空格键按下提笔
            self.can_draw[None] = 1
            self.bifeng[None] = 0

        if self.gui.is_pressed('q'):  # 按字母键q无级增大笔尖大小
            self.brush.pensize_init[None]+=0.06

        if self.gui.is_pressed('w'):  # 按字母键w无级减小笔尖大小
            self.brush.pensize_init[None]-=0.2

        if self.gui.is_pressed('r'):  # 按字母键重置笔尖大小
               self.brush.pensize_init[None]=10.0

        if self.can_draw[None] == 1 and self.gui.is_pressed('c'):  # 按字母键c重启程序（=清屏）
            python = sys.executable
            os.execl(python, python, *sys.argv)
  
        if self.gui.is_pressed(ti.GUI.ESCAPE):  # 按escape键退出程序
            sys.exit()

        if self.gui.is_pressed('s'):
            pre_pixels=self.canvas.pixels.to_numpy()
            pixels=np.zeros(   ( len(pre_pixels[0]),len(pre_pixels),3 )   )
            for i in range(len(pre_pixels[0])):
                for j in range(len(pre_pixels)):
                    for k in range(3):
                        pixels[len(pre_pixels[0])-1-i][j][k]=255.0*pre_pixels[j][i][k]
            
            now = datetime.datetime.now()
            #cv2.imwrite("./CanvasShot/"+now.strftime("%d_%H_%M_%S")+".png",pixels)
            #im.imsave("./CanvasShot/" + now.strftime("%d_%H_%M_%S")+".png", pixels/255)
            self.pic_order+=1


        self.brush.get_events(self.can_draw[None],self.bifeng[None])

    # 获取鼠标位置
    def gui_pos(self):
        pos = self.gui.get_cursor_pos()
        self.brush.get_pos(pos)
        self.brush.general_S()

    @ti.kernel
    # S值传递与运算
    def render(self):
        for i,j in ti.ndrange(self.res_x,self.res_y):
            self.canvas.sources[i,j]=self.brush.S[i,j]
        self.canvas.Update()

    # 定义panel运行
    def run(self):
        ss = 0
        while self.gui.running:
            ss += 1
            self.gui_judge()
            self.gui_pos()
            self.render()
            self.gui.set_image(self.canvas.showPic)#我tm真马勒为什么输出的PIXEL是有红色田字格的，但是set_image显示的就没有？
            #print(self.canvas.showPic)
            self.gui.show()
            # name = f"./result/RES_{ss:04d}.png"
            # self.gui.show(name)
        
    
