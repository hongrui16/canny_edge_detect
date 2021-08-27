import scipy
import numpy
from PIL import Image
import math
from math import pi
import imageio
import cv2
import numpy as np
import argparse
import os
import sys
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.ndimage.filters import convolve
from scipy.special import softmax
import time

def fwrite(txt_filepath, a_list):
    textfile = open(txt_filepath, "w")
    for element in a_list:
        textfile.write(str(element) + "\n")
    textfile.close()

def plot_and_save_fun(res,  mask_name = None, out_img_filepath = None):
    print(f'call {sys._getframe().f_code.co_name}')

    if len(res) <= 4:
        row = 1
        col = len(res)
    else:
        row = len(res)//4 + 1
        col = 4
    
    for i in range(len(res)):
        if i == 0:
            plt.subplot(row, col, i+1), plt.imshow(res[i].astype(np.uint8)), plt.title('ori'), plt.xticks([]), plt.yticks([])
        else:
            if mask_name:
                plt.subplot(row, col, i+1), plt.imshow(res[i].astype(np.uint8)), plt.title(mask_name[i]), plt.xticks([]), plt.yticks([])
            else:    
                plt.subplot(row, col, i+1), plt.imshow(res[i].astype(np.uint8)), plt.title(f'res_{i}'), plt.xticks([]), plt.yticks([])
        if res[i].ndim == 2:
            plt.gray()
    plt.subplots_adjust(wspace=0)
    if out_img_filepath:
        print(f'saving {out_img_filepath}')
        figure = plt.gcf()  
        figure.set_size_inches(16, 9)
        plt.savefig(out_img_filepath, dpi=900, bbox_inches='tight')
    plt.show()

def gaussian_filter_gray(gray, sigma = 2.2):
    print(f'call {sys._getframe().f_code.co_name}')

    gray_blur = ndimage.filters.gaussian_filter(gray, sigma)                           #gaussian low pass filter
    
    print(f'end of {sys._getframe().f_code.co_name}')

    return gray_blur

def sobel_filters(img):
    '''
    input:
        img: narray, blured image, 0~255
    output:
        Grad: narray, Grad, 0~255
        theta: [−π,π]
    '''
    print(f'call {sys._getframe().f_code.co_name}')
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    Grad = np.hypot(Ix, Iy)
    Grad = Grad / Grad.max() * 255
    theta = np.arctan2(Iy, Ix).astype(np.float32) #[−π,π]
    # angle = theta * 180. / np.pi
    # angle[angle < 0] += 180
    # return (Grad.astype(np.uint8), angle)
    print(f'end of {sys._getframe().f_code.co_name}')

    return (Grad.astype(np.uint8), theta)

def non_max_suppression(img, theta, debug = False):
    '''
    input:
        img: narray, Grad, 0~255
        theta: [−π,π]
    output:
        Z: narray, Grad after nms, 0~255
    '''
    # if debug:
    #     img[0] = 0
    #     img[-1] = 0
    #     img[:,0] = 0
    #     img[:,-1] = 0
    #     return img
    print(f'call {sys._getframe().f_code.co_name}')
    start = time.time()
    M, N = img.shape
    img = img.astype(np.uint8)
    Z = np.zeros((M,N), dtype=np.uint8)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    angle = angle.astype(np.float32)

    # for i in range(1,M-1):
    #     for j in range(1,N-1):
    #         try:
    #             q = 255
    #             r = 255

    #            #angle 0
    #             if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
    #                 q = img[i, j+1]
    #                 r = img[i, j-1]
    #             #angle 45
    #             elif (22.5 <= angle[i,j] < 67.5):
    #                 q = img[i+1, j-1]
    #                 r = img[i-1, j+1]
    #             #angle 90
    #             elif (67.5 <= angle[i,j] < 112.5):
    #                 q = img[i+1, j]
    #                 r = img[i-1, j]
    #             #angle 135
    #             elif (112.5 <= angle[i,j] < 157.5):
    #                 q = img[i-1, j-1]
    #                 r = img[i+1, j+1]

    #             if (img[i,j] >= q) and (img[i,j] >= r):
    #                 Z[i,j] = img[i,j]
    #             else:
    #                 Z[i,j] = 0
    #         except IndexError as e:
    #             pass

    angle[(0 <= angle) * (angle< 22.5)] = 0
    angle[(157.5 <= angle) * (angle < 180)] = 0
    angle[(22.5 <= angle) * (angle < 67.5)] = 45
    angle[(67.5 <= angle) * (angle < 112.5)] = 90
    angle[(112.5 <= angle) * (angle < 157.5)] = 135

    for i in range(1,M-1):
        for j in range(1,N-1):
            where = angle[i, j]
            try:
                if where == 0:
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        Z[i,j] = img[i,j]
                elif where == 90:
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                        Z[i,j] = img[i,j]
                elif where == 135:
                    if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
                        Z[i,j] = img[i,j]
                elif where == 45:
                    if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                        Z[i,j] = img[i,j]
            except IndexError as e:
                """ Todo: Deal with pixels at the image boundaries. """
                pass
    print(f'end of {sys._getframe().f_code.co_name}, costs {round(time.time()-start, 2)} s')
    return Z



def double_threshold(nms_grad, high_ratio = 0.15, low_ratio = 0.05 ):
    '''
    input:
        nms_grad: narray, 0~255
        high_ratio: high threshold ratio
        low_ratio: low threshold ratio
    output:
        strong_edge: narray, high threshold ~ 255
        weak_edge: narray, low threshold ~ high threshold
    '''
    print(f'call {sys._getframe().f_code.co_name}')
    start = time.time()

    # height, width = nms_grad.shape

    grad_peak = np.max(nms_grad)
    high_thres = high_ratio*grad_peak
    low_thres = low_ratio*grad_peak

    # strong_edge = np.zeros((height, width), dtype=np.uint8)
    # weak_edge = np.zeros((height, width), dtype=np.uint8)
    # for x in range(height):
    #     for y in range(width):
    #         if nms_grad[x][y]>=high_thres:
    #             strong_edge[x][y]=nms_grad[x][y]
    #         elif nms_grad[x][y]>=low_thres:
    #             weak_edge[x][y]=nms_grad[x][y]
    #         else:
    #             pass
    
    strong_edge_mask = nms_grad>=high_thres
    weak_edge_mask = (nms_grad>=low_thres)*(nms_grad<high_thres)
    strong_edge = nms_grad*strong_edge_mask.copy()
    weak_edge = nms_grad*weak_edge_mask.copy()

    print(f'end of {sys._getframe().f_code.co_name}, costs {round(time.time()-start, 2)} s')
    return strong_edge, weak_edge
    
def track_edge(strong_edge, weak_edge):
    '''
    input:
        strong_edge: narray, high threshold ~ 255
        weak_edge: narray, low threshold ~ high threshold
    output:
        strong_edge: narray, 0, 255
    '''
    print(f'call {sys._getframe().f_code.co_name}')
    start = time.time()

    def traverse(i, j):
        # print(f'{i} {j}')
        x = [-1, 0, 1, -1, 1, -1, 0, 1]
        y = [-1, -1, -1, 0, 0, 1, 1, 1]
        for k in range(8):
            if strong_edge[i+x[k]][j+y[k]]==0 and weak_edge[i+x[k]][j+y[k]]!=0:
                strong_edge[i+x[k]][j+y[k]]=255
                traverse(i+x[k], j+y[k])
    height, width = strong_edge.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            if strong_edge[i][j]:
                strong_edge[i][j]=255
                traverse(i, j)

    
    print(f'end of {sys._getframe().f_code.co_name}, costs {round(time.time()-start, 2)} s')
    return strong_edge

def hysteresis(strong_edge, weak_edge):
    '''
    input:
        strong_edge: narray, high threshold ~ 255
        weak_edge: narray, low threshold ~ high threshold
    output:
        strong_edge: narray, 0, 255
    '''
    print(f'call {sys._getframe().f_code.co_name}')
    start = time.time()

    M, N = strong_edge.shape
    weak = 0
    strong = strong_edge

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    print(f'end of {sys._getframe().f_code.co_name}, costs {round(time.time()-start, 2)} s')
    return strong
    
def delete_short_line(strong_edge, min_cnt):
    '''
    input:
        strong_edge: narray, 0, 1
        min_cnt: int, threshold
    output:
        strong_edge: narray, 0 1
    '''
    print(f'call {sys._getframe().f_code.co_name}')
    start = time.time()


    def traverse(i, j):
        # visited_pixel[i][j] = True 
        global_visited_pixel[i][j] = True        
        cnt[0] += 1
        # print(f'{i} {j}, cnt:{cnt[0]}')
        x = [-1, 0, 1, -1, 1, -1, 0, 1]
        y = [-1, -1, -1, 0, 0, 1, 1, 1]
        for k in range(8):
            if strong_edge[i+x[k]][j+y[k]] == 255 and visited_pixel[i+x[k]][j+y[k]] == False:
                visited_pixel[i+x[k]][j+y[k]] = True
                traverse(i+x[k], j+y[k])

    height, width = strong_edge.shape
    global_visited_pixel = np.full((height, width), False, dtype=bool)
    for i in range(1, height-1):
        for j in range(1, width-1):
            # if strong_edge[i][j] and not visited_pixel[i][j]:
            #     visited_pixel[i][j] = True
            #     cnt = np.zeros(1)
            #     traverse(i, j)
            #     if cnt[0] < min_cnt and len(visited_pixel==True) > 0:
            #         strong_edge[visited_pixel] = 0
            if strong_edge[i][j] and not global_visited_pixel[i][j]:
                global_visited_pixel[i][j] = True
                cnt = np.zeros(1)
                visited_pixel = np.full((height, width), False, dtype=bool)
                traverse(i, j)
                if cnt[0] < min_cnt and len(visited_pixel[visited_pixel==True]) > 0:
                    strong_edge[visited_pixel] = 0
            

    print(f'end of {sys._getframe().f_code.co_name}, costs {round(time.time()-start, 2)} s')
    return strong_edge

def delete_free_form_curve_by_std(edge, theta, avg_angle_limit = 10, std_angle_thres = 20):
    '''
    input:
        edge: narray, 0, 255
        theta: [−π,π]
    output:
        edge: narray, 0, 255
    '''
    print(f'call {sys._getframe().f_code.co_name}')
    start = time.time()

    angle = theta.copy() * 180. / np.pi
    angle[angle < 0] += 360
    def traverse(i, j):
        # visited_pixel[i][j] = True 
        global_visited_coors[i][j] = True        
        x = [-1, 0, 1, -1, 1, -1, 0, 1]
        y = [-1, -1, -1, 0, 0, 1, 1, 1]
        for k in range(8):
            if edge[i+x[k]][j+y[k]] == 255 and visited_coors[i+x[k]][j+y[k]] == False:
                visited_coors[i+x[k]][j+y[k]] = True
                traverse(i+x[k], j+y[k])

    height, width = edge.shape
    global_visited_coors = np.full((height, width), False, dtype=bool)
    for i in range(1, height-1):
        for j in range(1, width-1):
            if edge[i][j] and not global_visited_coors[i][j]:
                global_visited_coors[i][j] = True
                visited_coors = np.full((height, width), False, dtype=bool)
                traverse(i, j)
                # avg_angle = np.mean(angle[visited_coors])
                std_angle = np.std(angle[visited_coors])
                if std_angle > std_angle_thres:
                    edge[visited_coors] = 0
            

    print(f'end of {sys._getframe().f_code.co_name}, costs {round(time.time()-start, 2)} s')
    return edge

def delete_free_form_curve_by_entropy(edge, theta, entropy_thres = 0.008):
    '''
    input:
        edge: narray, 0, 255
        theta: [−π,π]
    output:
        edge: narray, 0, 255
    '''
    print(f'call {sys._getframe().f_code.co_name}')
    start = time.time()


    entropy_logfile = 'entropy.txt'
    if os.path.exists(entropy_logfile):
        os.remove(entropy_logfile)

    angle = theta.copy() * 180. / np.pi
    angle[angle < 0] += 360
    angle_quant = np.zeros(angle.shape, dtype=np.uint8)
    # print('angle_quant', angle_quant.shape)
    block_cnt = 16
    block_size = 360/block_cnt
    for i in range(block_cnt):
        left = i*block_size
        right = (i+1)*block_size
        angle_quant[(angle >=left) * (angle < right)] = i
    # print('angle_quant', angle_quant)
    def traverse(i, j):
        # visited_pixel[i][j] = True 
        global_visited_coors[i][j] = True        
        x = [-1, 0, 1, -1, 1, -1, 0, 1]
        y = [-1, -1, -1, 0, 0, 1, 1, 1]
        for k in range(8):
            if edge[i+x[k]][j+y[k]] == 255 and visited_coors[i+x[k]][j+y[k]] == False:
                visited_coors[i+x[k]][j+y[k]] = True
                traverse(i+x[k], j+y[k])

    entropy_lists = []
    height, width = edge.shape
    angle_dist_cnt = np.zeros(block_cnt)
    global_visited_coors = np.full((height, width), False, dtype=bool)
    for i in range(1, height-1):
        for j in range(1, width-1):
            if edge[i][j] and not global_visited_coors[i][j]:
                global_visited_coors[i][j] = True
                visited_coors = np.full((height, width), False, dtype=bool)
                traverse(i, j)
                # avg_angle = np.mean(angle[visited_coors])
                selected_line_angle = angle_quant[visited_coors]
                if len(selected_line_angle) <= 0:
                    continue
                # print('selected_line_angle', selected_line_angle)
                for idx in range(block_cnt):
                    # print(f'len(selected_line_angle=={idx})', len(selected_line_angle[selected_line_angle==idx]))
                    angle_dist_cnt[idx] = len(selected_line_angle[selected_line_angle==idx])
                # print('angle_dist_cnt', angle_dist_cnt)  
                entropy_lists.append(angle_dist_cnt.tolist())
                # angle_dist_cnt[angle_dist_cnt==0] = 1
                total_num = np.sum(angle_dist_cnt)
                angle_dist_cnt /= total_num
                angle_dist_prob = softmax(angle_dist_cnt)
                entropy_lists.append(angle_dist_prob.tolist())
                entropy = -1.*np.dot(angle_dist_prob, np.log(angle_dist_prob))/total_num
                entropy_lists.append(entropy)
                if entropy > entropy_thres:
                    edge[visited_coors] = 0

    fwrite(entropy_logfile, entropy_lists)

    print(f'end of {sys._getframe().f_code.co_name}, costs {round(time.time()-start, 2)} s')
    return edge

def open_alg_fun():
    res  = []
    mask_name = []
    img_filepath = 'D:\\test\\rail_road_detection\\Canny-Python\\outs2\\rail_05_edgeFiltered_1st.jpg'
    img = cv2.imread(img_filepath, 0)

    res.append(img.copy())   
    mask_name.append('ori') 


    设置卷积核
    kernel = np.ones((2, 2),np.uint8)
    erode = cv2.erode(img, kernel, iterations = 1)
    res.append(erode.copy())   
    mask_name.append('erode') 

    kernel = np.ones((2, 2),np.uint8)
    dilate = cv2.dilate(img, kernel, iterations = 1)
    res.append(dilate.copy())   
    mask_name.append('dilate') 

    #图像开运算
    kernel = np.ones((2, 2),np.uint8)

    dst = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    res.append(dst.copy())   
    mask_name.append('res') 
    plot_and_save_fun(res, mask_name)#, out_img_filepath)

def main(args):
    img_filepath = args.img_filepath
    output_dir = args.output_dir

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    debug = True
    sigma = 2.2
    res  = []
    mask_name = []
    # f = 'Lenna.png'
    f = img_filepath
    img_name = f.split('\\')[-1] if '\\' in f else f
    img_name_prefix = img_name.split('.')[0]

    img = Image.open(f).convert('L')                                          #grayscale
    print('img.size', img.size)
    sys.setrecursionlimit(img.size[0]*img.size[1]) ### set maximum recursion depth

    img = np.array(img, dtype = np.uint8)
    res.append(img.copy())   
    if output_dir:
        imageio.imwrite(os.path.join(output_dir, f'{img_name_prefix}_ori.jpg'), img)
    mask_name.append('ori')     

    img_blur = gaussian_filter_gray(img.copy().astype(np.float32), sigma)                           #gaussian low pass filter

    sobel_grad, sobel_theta = sobel_filters(img_blur)
    res.append(sobel_grad.copy())                                 
    res.append(sobel_theta.copy())                                 
    if output_dir:
        imageio.imwrite(os.path.join(output_dir, f'{img_name_prefix}_grad.jpg'), sobel_grad)
        imageio.imwrite(os.path.join(output_dir, f'{img_name_prefix}_angle.jpg'), sobel_theta)
    mask_name.append('grad')     
    mask_name.append('angle')     
    

    nms_grad = non_max_suppression(sobel_grad, sobel_theta.copy(), debug)
    res.append(nms_grad.copy())                                 
    if output_dir:
        imageio.imwrite(os.path.join(output_dir, f'{img_name_prefix}_nmsGrad.jpg'), nms_grad)
    mask_name.append('nmsGrad')     



    strong_edge, weak_edge = double_threshold(nms_grad)
    res.append(weak_edge.copy())                                 
    res.append(strong_edge.copy())                                 
    if output_dir:
        imageio.imwrite(os.path.join(output_dir, f'{img_name_prefix}_weakEdge.jpg'), weak_edge)
        imageio.imwrite(os.path.join(output_dir, f'{img_name_prefix}_strongEdge.jpg'), strong_edge)
    mask_name.append('weakEdge')     
    mask_name.append('strongEdge')     

    canny_edge = track_edge(strong_edge, weak_edge)
    res.append(canny_edge.copy())                                 
    # imageio.imwrite('cannynewout.jpg', strong_edge)
    if output_dir:
        imageio.imwrite(os.path.join(output_dir, f'{img_name_prefix}_cannyEdge.jpg'), canny_edge)
    mask_name.append('cannyEdge')     
    

    edge_filtered_1st = delete_short_line(canny_edge.copy(), 100)
    res.append(edge_filtered_1st.copy())                                 
    if output_dir:
        imageio.imwrite(os.path.join(output_dir, f'{img_name_prefix}_edgeFiltered_1st.jpg'), edge_filtered_1st)
    mask_name.append('edgeFiltered_1st')   

    edge_minus_1st = np.subtract(canny_edge.copy(), edge_filtered_1st.copy())
    res.append(edge_minus_1st.copy())                                 
    if output_dir:
        imageio.imwrite(os.path.join(output_dir, f'{img_name_prefix}_edgeMinus_1st.jpg'), edge_minus_1st)
    mask_name.append('edgeMinus_1st')   
    
    # if debug:
    #     img_filepath = 'D:\\test\\rail_road_detection\\Canny-Python\\outs2\\rail_05_edgeFiltered_1st.jpg'
    #     edge_filtered_1st = cv2.imread(img_filepath, 0)
    # edge_filtered_2nd = delete_free_form_curve(edge_filtered_1st.copy(), sobel_theta.copy(), 10, 70)
    edge_filtered_2nd = delete_free_form_curve_by_entropy(edge_filtered_1st.copy(), sobel_theta)
    res.append(edge_filtered_2nd.copy())                                 
    if output_dir:
        imageio.imwrite(os.path.join(output_dir, f'{img_name_prefix}_edgeFiltered_2nd.jpg'), edge_filtered_2nd)
    mask_name.append('edgeFiltered_2nd')   

    edge_minus_2nd = np.subtract(edge_filtered_1st.copy(), edge_filtered_2nd.copy())
    res.append(edge_minus_2nd.copy())                                 
    if output_dir:
        imageio.imwrite(os.path.join(output_dir, f'{img_name_prefix}_edgeMinus_2nd.jpg'), edge_minus_2nd)
    mask_name.append('edgeMinus_2nd')   

    out_img_filepath = os.path.join(output_dir, img_name)
    plot_and_save_fun(res, mask_name, out_img_filepath)
            


# def binary_search(Q, lists, L, R):
#     length = len(lists[L:R])
#     if length == 1:
#         if Q == lists[L]:
#             print(L)
#         else:
#             print(-1)

#     elif Q < lists[(L+R)//2]:
#         binary_search(Q, lists, L, (L+R)//2)

#     else:
#         binary_search(Q, lists, (L+R)//2, L)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='aligment')
    parser.add_argument('-if', '--img_filepath', type=str, default='images\\rail_06.jpg')

    parser.add_argument('-im', '--input_dir', type=str, default=None)

    parser.add_argument('-om', '--output_dir', type=str, default=None)
    parser.add_argument('-th', '--target_height', type=int, default=1440)
    parser.add_argument('-tw', '--target_width', type=int, default=1920)

    args = parser.parse_args()
    main(args)
    # open_alg_fun()

    # cal_Pi()
    # lists = [0,1,2,4,5,6]
    # res = binary_search(3,lists, 0, 6)
    # print(res)
    # res = binary_search(4,lists, 0, 6)

    # lists = [1,2,4,5,6]
    # res = binary_search(3,lists, 0, 6)
    # print(res)
    # res = binary_search(4,lists, 0, 6)
