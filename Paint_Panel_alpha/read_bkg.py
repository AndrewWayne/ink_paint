import matplotlib.image as im
import numpy as np
image_path="./fibre.png"
def get_bkg(image_path):
    # RGB图 黑色是0，白色是1（也不知道这是什么道理）
    fibre = im.imread(image_path)
    res_x = len(fibre)
    res_y = len(fibre[0])
    bkg = np.zeros((res_x+2,res_y+2, 3))

    for x in range(res_x+2):
        for y in range(res_y+2):
            if x==0 or x==res_x+1 or y==0 or y==res_y+1:
                bkg[x, y] = [0, 0, 0]
            else:
                bkg[x, y] = [255, 255, 255] - fibre[x-1][y-1][0:3]*255 - [1, 1, 1]
                bkg[x, y, 0] = max(0, bkg[x, y, 0])
                bkg[x, y, 1] = max(0, bkg[x, y, 1])
                bkg[x, y, 2] = max(0, bkg[x, y, 2])
    
    return bkg

def D_map(white_depth):  # 这个函数将颜色的“灰度”映射到扩散系数。注意：黑色时扩散系数应该最大，白色时应该最小。
    d = (1.2-(0.83*white_depth))*0.106
    return d

