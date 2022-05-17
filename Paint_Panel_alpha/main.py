import taichi as ti
import Panel
from read_fibre import *

ti.init(arch=ti.gpu)

# 获取纤维图
fibre_path="./fibre.png"
fibre = get_D(fibre_path)

# 调用面板
panel=Panel.Panel(fibre)
panel.run()
