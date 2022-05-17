import matplotlib.image as im
import numpy as np
image_path="./fibre.png"
def get_D(image_path):
    # RGB图 黑色是0，白色是1（也不知道这是什么道理）
    fibre = im.imread(image_path)
    res_x = len(fibre)
    res_y = len(fibre[0])
    edged_fibre = np.zeros((res_x+2,res_y+2))
    D = np.zeros((res_x+2,res_y+2))

    for x in range(res_x+2):
        for y in range(res_y+2):
            if x==0 or x==res_x+1 or y==0 or y==res_y+1:
                edged_fibre[x][y]=0
            else:
                edged_fibre[x][y]=fibre[x-1][y-1][0]

    # 扩散系数越大，扩散越快，D 越大，常见液体的扩散系数应该在 1 左右。
    for x in range(res_x+2):
        for y in range(res_y+2):
            D[x,y]=D_map(edged_fibre[x][y])
    
    return D

def D_map(white_depth):  # 这个函数将颜色的“灰度”映射到扩散系数。注意：黑色时扩散系数应该最大，白色时应该最小。
    d = (1.2-(0.83*white_depth))*0.106
    return d

