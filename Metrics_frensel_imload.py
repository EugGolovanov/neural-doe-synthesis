import numpy as np


# ПАРАМЕТРЫ (метры)
LAM = 532.0e-9       
dx = 10.0e-6          
Z_1 = 0.0241 
Z_2 = 0.0253
Z_3 = 0.0265
Z_3 = 0.0277 


def normalization(array, dynamic_range):
    return array / np.max(array) * dynamic_range


def correlation_coef(base_image, new_image, dynamic_range=255):
    """
    Предварительная нормировка и разгон до 255
    """
    base_image_normalized = normalization(base_image, dynamic_range)
    new_image_normalized = normalization(new_image, dynamic_range)
    return np.corrcoef(base_image_normalized.flatten(), new_image_normalized.flatten())[0, 1]


def ssim(base_image, new_image, dynamic_range=255):
    """
    Изменяется от -1 до +1, чем ближе к +1 тем изображения похожее
    Предварительная нормировка и разгон до 255
    """
    base_image_normalized = normalization(base_image, dynamic_range)
    new_image_normalized = normalization(new_image, dynamic_range)
    mean_base = np.mean(base_image_normalized)
    mean_new = np.mean(new_image_normalized)
    std_base = np.std(base_image_normalized)
    std_new = np.std(new_image_normalized)
    cov = (np.mean(base_image_normalized*new_image_normalized)
           - mean_base*mean_new)
    c1 = (0.01 * dynamic_range)**2
    c2 = (0.03 * dynamic_range)**2
    ssim_value = ((2*mean_base*mean_new + c1) * (2*cov + c2)
                  / ((mean_base**2 + mean_new**2 + c1)
                     * (std_base**2 + std_new**2 + c2)))
    return ssim_value


# задание координатной сетки в плоскости голограммы
# dx - размер пикселя модулятора

def create_grids(shape, dx):
    global X1, Y1
    h, w = shape
    x1 = (np.arange(w) - w // 2) * dx
    y1 = (np.arange(h) - h // 2) * dx
    X1, Y1 = np.meshgrid(x1, y1)

# Прямое расспространение света по фрмуле дифракции Френеля (без двух экспонент перед фурье)
def forward_propagation(field, lam, z):
    k = 2.0 * np.pi / lam
    Q1 = np.exp(1j * k * (X1 ** 2 + Y1 ** 2) / (2.0 * z))
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field * Q1)))
    return spectrum

