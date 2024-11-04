import numpy as np
import math
import skimage as ski
from skimage.color import rgb2gray


# ╭──────────────────────────────────────────────────────────╮
# │                       Open Images                        │
# ╰──────────────────────────────────────────────────────────╯

def open_image(path):
    image = ski.io.imread(path)
    return image


def open_gray_scale(path):
    image = ski.io.imread(path)
    image = rgb2gray(image)
    return image


def to_gray_scale(image):
    height = image.shape[0]
    width = image.shape[1]
    gray_scale_image = np.empty((height, width), int)

    for h in range(height):
        for w in range(width):
            gray_scale_image[h][w] = math.floor((
                    image[h][w][0] * 0.2989 +
                    image[h][w][1] * 0.5870 +
                    image[h][w][2] * 0.1140))

    return gray_scale_image


# ╭──────────────────────────────────────────────────────────╮
# │                     Equalize Images                      │
# ╰──────────────────────────────────────────────────────────╯

def equalize_image_path(path, no_levels):
    
    image = open_image(path)
    image = to_gray_scale(image)
    height = image.shape[0]
    width = image.shape[1]
    histogram = histogram_probabilities_sum(image, no_levels)

    img_out = np.empty((height, width), int)
    for h in range(height):
        for w in range(width):
            img_out[h][w] = (no_levels - 1) * histogram[image[h][w]]

    return img_out


def equalize_image_ndarray(image, no_levels):
    image = to_gray_scale(image)
    height = image.shape[0]
    width = image.shape[1]
    histogram = histogram_probabilities_sum(image, no_levels)

    img_out = np.empty((height, width), int)
    for h in range(height):
        for w in range(width):
            img_out[h][w] = (no_levels - 1) * histogram[image[h][w]]

    return img_out


def equalize_image_lut(image, lut):
    # image = to_gray_scale(image)
    height = image.shape[0]
    width = image.shape[1]

    img_out = np.empty((height, width), int)
    for h in range(height):
        for w in range(width):
            img_out[h][w] = (len(lut) - 1) * lut[image[h][w]]

    return img_out


# ╭──────────────────────────────────────────────────────────╮
# │                     Trans Functions                      │
# ╰──────────────────────────────────────────────────────────╯

def trans_trozos1(image, r1):
    height = image.shape[0]
    width = image.shape[1]

    img_out = np.empty((height, width), int)
    for h in range(height):
        for w in range(width):
            if image[h][w] < r1:
                img_out[h][w] = 0
            else:
                img_out[h][w] = 255

    return img_out


def trans_trozos2(image, r1, r2):
    height = image.shape[0]
    width = image.shape[1]

    img_out = np.empty((height, width), int)
    for h in range(height):
        for w in range(width):
            if image[h][w] < r1:
                img_out[h][w] = 0
            elif image[h][w] < r2:
                img_out[h][w] = 128
            else:
                img_out[h][w] = 255

    return img_out


# ╭──────────────────────────────────────────────────────────╮
# │                      LUT Functions                       │
# ╰──────────────────────────────────────────────────────────╯

def LUT_trozos(int_max, r1, r2):
    LUT = np.empty(int_max + 1, int)
    for i in range(int_max + 1):
        if i < r1:
            LUT[i] = 0
        elif i < r2:
            LUT[i] = 128
        else:
            LUT[i] = 255
    return LUT


def trans_trozos_LUT(image, LUT):
    height = image.shape[0]
    width = image.shape[1]

    img_out = np.empty((height, width), int)
    for h in range(height):
        for w in range(width):
            img_out[h][w] = LUT[int(image[h][w])]

    return img_out


# ╭──────────────────────────────────────────────────────────╮
# │                        Cuantizar                         │
# ╰──────────────────────────────────────────────────────────╯

def cuantizar_img(image, bit_number):
    height = image.shape[0]
    width = image.shape[1]

    img_out = np.empty((height, width), int)
    for h in range(height):
        for w in range(width):
            img_out[h][w] = math.floor(((image[h][w] / 255) * (pow(2, bit_number))))

    return img_out


# ╭──────────────────────────────────────────────────────────╮
# │                       Histogramas                        │
# ╰──────────────────────────────────────────────────────────╯

def histogram_count(image, no_levels):
    height = image.shape[0]
    width = image.shape[1]
    histogram = [0] * no_levels

    for h in range(height):
        for w in range(width):
            histogram[image[h][w]] += 1

    histogram = np.array(histogram)

    return histogram


def histogram_probabilities(image, no_levels):
    height = image.shape[0]
    width = image.shape[1]
    histogram = [0] * no_levels

    for h in range(height):
        for w in range(width):
            histogram[image[h][w]] += 1

    for i in range(no_levels):
        histogram[i] = histogram[i] / (height * width)

    histogram = np.array(histogram)

    return histogram


def histogram_probabilities_sum(image, no_levels):
    histogram = histogram_probabilities(image, no_levels)
    for i in range(1, len(histogram)):
        histogram[i] += histogram[i - 1]

    return histogram


# ╭──────────────────────────────────────────────────────────╮
# │                       Convolucion                        │
# ╰──────────────────────────────────────────────────────────╯

def convolucion_scipy(image, kernel):
    kernel_length = len(kernel)

    image = np.pad(image, math.floor(kernel_length/2), constant_values=0)

    height = image.shape[0]
    width = image.shape[1]

    center_offset = math.floor(kernel_length/2)

    img_out = np.empty((height, width), int)
    for h in range(height):
        for w in range(width):
            pixel_value = 0

            for i in range(-center_offset, center_offset+1):
                for j in range(-center_offset, center_offset+1):
                    image_pixel_value = 0

                    if (not w+j < 0 and not w+j >= width) and (not h+i < 0 and not h+i >= height):
                        image_pixel_value = image[h+i][w+j]
                    pixel_value += image_pixel_value * kernel[i+center_offset][j+center_offset]

            img_out[h][w] = int(pixel_value)
            # out_string = '┏' + ('━━━┳' * (len(values) - 1)) + '━━━┓'
            # for i in range(len(values)):
            #     out_string += '\n'
            #     for j in range(len(values)):
            #         if j == 0:
            #             out_string += '┃'
            #         out_string += str(values[i][j]).ljust(3, ' ') + '┃'
            #
            #     if i == math.floor((len(values) / 2)):
            #         out_string += ' = %s' % (pixel_value)
            #
            #     if not i == (len(values) - 1):
            #         out_string += '\n┣' + ('━━━╋' * (len(values) - 1)) + '━━━┫'
            # out_string += '\n┗' + ('━━━┻' * (len(values) - 1)) + '━━━┛'
            # print(out_string)
            # return
    return img_out


# ╭──────────────────────────────────────────────────────────╮
# │                     Media Aritmetica                     │
# ╰──────────────────────────────────────────────────────────╯

def media_aritmetica(image, w_size):
    height = image.shape[0]
    width = image.shape[1]

    center_offset = math.floor(w_size/2)

    img_out = np.empty((height, width), np.float64)

    for h in range(height):
        for w in range(width):
            pixel_value = 0

            for i in range(-center_offset, center_offset+1):
                for j in range(-center_offset, center_offset+1):
                    image_pixel_value = 0

                    outside_negative_width = w+j < 0
                    outside_positive_width = w+j >= width

                    outside_negative_height = h+i < 0
                    outside_positive_height = h+i >= height

                    outside_width = (outside_negative_width
                                     or
                                     outside_positive_width)

                    outside_height = (outside_negative_height
                                      or
                                      outside_positive_height)

                    # if (not w+j < 0 and not w+j >= width) and (not h+i < 0 and not h+i >= height):
                    if (not outside_width) and (not outside_height):
                        image_pixel_value = image[h+i][w+j]

                    pixel_value += image_pixel_value

            img_out[h][w] = pixel_value/(w_size*w_size)

    return img_out


def media_geometrica(image, w_size):
    height = image.shape[0]
    width = image.shape[1]

    center_offset = math.floor(w_size/2)

    img_out = np.zeros((height, width), np.float64)

    for h in range(height):
        for w in range(width):
            pixel_value = 1
            # values = [[0 for _ in range(w_size)] for _ in range(w_size)]

            for i in range(-center_offset, center_offset+1):
                for j in range(-center_offset, center_offset+1):
                    image_pixel_value = 0
                    outside_negative_width = w+j < 0
                    outside_positive_width = w+j >= width

                    outside_negative_height = h+i < 0
                    outside_positive_height = h+i >= height

                    outside_width = (outside_negative_width
                                     or
                                     outside_positive_width)

                    outside_height = (outside_negative_height
                                      or
                                      outside_positive_height)

                    if (not outside_width) and (not outside_height):
                        image_pixel_value = image[h+i][w+j]

                    pixel_value = np.multiply(pixel_value,
                                              image_pixel_value,
                                              dtype=object)

                    # values[i+center_offset][j+center_offset] = (
                    #         image_pixel_value)

            # out_string = '┏' + ('━━━┳' * (len(values) - 1)) + '━━━┓'
            # for i in range(len(values)):
            #     out_string += '\n'
            #     for j in range(len(values)):
            #         if j == 0:
            #             out_string += '┃'
            #         out_string += str(values[i][j]).ljust(3, ' ') + '┃'
            #
            #     if i == math.floor((len(values) / 2)):
            #         out_string += ' = %s' % (pixel_value)
            #
            #     if not i == (len(values) - 1):
            #         out_string += '\n┣' + ('━━━╋' * (len(values) - 1)) + '━━━┫'
            # out_string += '\n┗' + ('━━━┻' * (len(values) - 1)) + '━━━┛'
            # # out_string += f'= {pixel_value}\n\n'
            # print(out_string)
            img_out[h][w] = pixel_value**(1/(w_size*w_size))

    return img_out



def media_armonico(image, w_size, epsilon):
    height = image.shape[0]
    width = image.shape[1]

    center_offset = math.floor(w_size/2)

    img_out = np.empty((height, width), np.float64)

    for h in range(height):
        for w in range(width):
            neighborhood_sum = 0

            for i in range(-center_offset, center_offset+1):
                for j in range(-center_offset, center_offset+1):
                    image_pixel_value = epsilon

                    outside_negative_width = w+j < 0
                    outside_positive_width = w+j >= width

                    outside_negative_height = h+i < 0
                    outside_positive_height = h+i >= height

                    outside_width = (outside_negative_width
                                     or
                                     outside_positive_width)

                    outside_height = (outside_negative_height
                                      or
                                      outside_positive_height)

                    if (not outside_width) and (not outside_height):
                        image_pixel_value = (image[h+i][w+j] + epsilon)

                    neighborhood_sum += image_pixel_value

            img_out[h][w] = ((w_size*w_size)/(1/neighborhood_sum))

    return img_out


def media_contra_harmonico(image, w_size, Q):
    height = image.shape[0]
    width = image.shape[1]

    center_offset = math.floor(w_size/2)

    img_out = np.empty((height, width), np.float64)

    for h in range(height):
        for w in range(width):
            neighborhood_sum = 0

            for i in range(-center_offset, center_offset+1):
                for j in range(-center_offset, center_offset+1):
                    image_pixel_value = 0

                    outside_negative_width = w+j < 0
                    outside_positive_width = w+j >= width

                    outside_negative_height = h+i < 0
                    outside_positive_height = h+i >= height

                    outside_width = (outside_negative_width
                                     or
                                     outside_positive_width)

                    outside_height = (outside_negative_height
                                      or
                                      outside_positive_height)

                    if (not outside_width) and (not outside_height):
                        image_pixel_value = image[h+i][w+j]

                    neighborhood_sum += image_pixel_value

            img_out[h][w] = math.pow((neighborhood_sum), (Q+1))/math.pow((neighborhood_sum), Q)

    return img_out


def media_mediana(image, w_size):
    height = image.shape[0]
    width = image.shape[1]

    center_offset = math.floor(w_size/2)

    img_out = np.empty((height, width), np.float64)

    for h in range(height):
        for w in range(width):
            neighborhood_sum = []

            for i in range(-center_offset, center_offset+1):
                for j in range(-center_offset, center_offset+1):
                    image_pixel_value = 0

                    outside_negative_width = w+j < 0
                    outside_positive_width = w+j >= width

                    outside_negative_height = h+i < 0
                    outside_positive_height = h+i >= height

                    outside_width = (outside_negative_width
                                     or
                                     outside_positive_width)

                    outside_height = (outside_negative_height
                                      or
                                      outside_positive_height)

                    if (not outside_width) and (not outside_height):
                        image_pixel_value = image[h+i][w+j]

                    neighborhood_sum.append(image_pixel_value)

            neighborhood_sum.sort()
            median = 0
            midpoint = (len(neighborhood_sum)/2) - 1
            if (len(neighborhood_sum) % 2 == 0):
                median = (neighborhood_sum[midpoint]+neighborhood_sum[midpoint])/2
            else:
                median = neighborhood_sum[math.ceil(midpoint)]


            img_out[h][w] = median

    return img_out


def filtro_maximo(image, w_size):
    height = image.shape[0]
    width = image.shape[1]

    center_offset = math.floor(w_size/2)

    img_out = np.empty((height, width), np.float64)

    for h in range(height):
        for w in range(width):
            neighborhood_sum = []

            for i in range(-center_offset, center_offset+1):
                for j in range(-center_offset, center_offset+1):
                    image_pixel_value = 0

                    outside_negative_width = w+j < 0
                    outside_positive_width = w+j >= width

                    outside_negative_height = h+i < 0
                    outside_positive_height = h+i >= height

                    outside_width = (outside_negative_width
                                     or
                                     outside_positive_width)

                    outside_height = (outside_negative_height
                                      or
                                      outside_positive_height)

                    if (not outside_width) and (not outside_height):
                        image_pixel_value = image[h+i][w+j]

                    neighborhood_sum.append(image_pixel_value)

            img_out[h][w] = max(neighborhood_sum)

    return img_out


def filtro_minimo(image, w_size):
    height = image.shape[0]
    width = image.shape[1]

    center_offset = math.floor(w_size/2)

    img_out = np.empty((height, width), np.float64)

    for h in range(height):
        for w in range(width):
            neighborhood_sum = []

            for i in range(-center_offset, center_offset+1):
                for j in range(-center_offset, center_offset+1):
                    image_pixel_value = 0

                    outside_negative_width = w+j < 0
                    outside_positive_width = w+j >= width

                    outside_negative_height = h+i < 0
                    outside_positive_height = h+i >= height

                    outside_width = (outside_negative_width
                                     or
                                     outside_positive_width)

                    outside_height = (outside_negative_height
                                      or
                                      outside_positive_height)

                    if (not outside_width) and (not outside_height):
                        image_pixel_value = image[h+i][w+j]

                    neighborhood_sum.append(image_pixel_value)

            img_out[h][w] = min(neighborhood_sum)

    return img_out


def filtro_alpha(image, w_size, alpha):
    height = image.shape[0]
    width = image.shape[1]

    center_offset = math.floor(w_size/2)

    img_out = np.empty((height, width), np.float64)

    for h in range(height):
        for w in range(width):
            neighborhood_sum = []

            for i in range(-center_offset, center_offset+1):
                for j in range(-center_offset, center_offset+1):
                    image_pixel_value = 0

                    outside_negative_width = w+j < 0
                    outside_positive_width = w+j >= width

                    outside_negative_height = h+i < 0
                    outside_positive_height = h+i >= height

                    outside_width = (outside_negative_width
                                     or
                                     outside_positive_width)

                    outside_height = (outside_negative_height
                                      or
                                      outside_positive_height)

                    if (not outside_width) and (not outside_height):
                        image_pixel_value = image[h+i][w+j]

                    neighborhood_sum.append(image_pixel_value)

            neighborhood_sum.sort()

            d = alpha*(w_size*w_size)

            del neighborhood_sum[0:math.ceil(d)]
            del neighborhood_sum[-(math.ceil(d)):]
            img_out[h][w] = (1/len(neighborhood_sum)) * sum(neighborhood_sum)

    return img_out
