import numpy as np
import cv2

def add_gaussian_noise(img_np:np.ndarray, level:float=0.1, mean:float=0, std:float=0.1) -> np.ndarray:
    '''
    input:
        img_np: should be a numpy array of uint8 or float32, can be greyscale or RGB
        level:  the percentage of pixels to be changed, level = 0.1 means that only 10% of the pixels will be changed
        mean:   the mean of the gaussian distribution
        std:    the standard deviation of the gaussian distribution
    output:
        noisy:  a numpy array of float32, range [0, 1]
    '''
    
    img_np = np.float32(img_np) / 255.0 if img_np.dtype == np.uint8 else img_np
    choice = np.random.choice([0, 1], size=img_np.shape, p=[1-level, level])
    noise  = np.random.normal(mean, std, img_np.shape)
    noisy_mask = np.where(choice, noise, 0)
    noisy  = img_np + noisy_mask
    noisy  = np.clip(noisy, 0, 1)
    return noisy


def add_salt_and_pepper_noise(img_np:np.ndarray, level:float=0.1) -> np.ndarray:
    '''
    input:
        img_np: should be a numpy array of uint8 or float32, can be greyscale or RGB
        level:  the percentage of pixels to be changed, level = 0.1 means that only 10% of the pixels will be changed
    output:
        noisy:  a numpy array of float32, range [0, 1]
    '''
    
    img_np = np.float32(img_np) / 255.0 if img_np.dtype == np.uint8 else img_np
    noisy_image = img_np.copy()
    level = level / 2
    choice = np.random.choice([0, 1, 2],  size=img_np.shape[:2], p=[1-level*2, level, level])

    noisy_image[choice == 1] = 1
    noisy_image[choice == 2] = 0
    
    return noisy_image

def add_speckle_noise(img_np:np.ndarray, level:float=0.1, mean:float=0, std:float=0.1):
    '''
    input:
        img_np: should be a numpy array of uint8 or float32, can be greyscale or RGB
        level:  the percentage of pixels to be changed, level = 0.1 means that only 10% of the pixels will be changed
        mean:   the mean of the gaussian distribution
        std:    the standard deviation of the gaussian distribution
    output:
        noisy:  a numpy array of float32, range [0, 1]
    '''
    img_np = np.float32(img_np) / 255.0 if img_np.dtype == np.uint8 else img_np
    speckle = np.random.normal(mean, std, img_np.shape)
    choice  = np.random.choice([0, 1], size=img_np.shape, p=[1-level, level])
    noisy_img = img_np + img_np * speckle * choice
    noisy_img = np.clip(noisy_img, 0, 1)
    return noisy_img

def add_poission_noise(img_np:np.ndarray, level:float=0.1, poi_level:float=1):
    '''
    input:
        img_np: should be a numpy array of uint8 or float32, can be greyscale or RGB
        level:  the percentage of pixels to be changed, level = 0.1 means that only 10% of the pixels will be changed
        poi_level: the level of poission noise, the higher the level, the more noise
    output:
        noisy:  a numpy array of float32, range [0, 1]
    '''
    # Add Poisson noise to the image
    img_np = np.float32(img_np) / 255.0 if img_np.dtype == np.uint8 else img_np
    choice = np.random.choice([0, 1], size=img_np.shape, p=[1-level, level])
    poisson_noise = np.random.poisson(img_np*255*poi_level) / 255
    
    noise_mask = choice * poisson_noise
    noisy_image = np.clip(img_np + noise_mask, 0, 1)
    return noisy_image


def motion_blur(img_np:np.ndarray, degree:int=0, angle:int=45):
    '''
    input:
        img_np: should be a numpy array of uint8 or float32, can be greyscale or RGB
        degree: the degree of the motion blur, the higher the degree, the more motion blur
        angle : the angle of the motion blur, range [0, 360]
    output:
        blurred:  a numpy array of float32, range [0, 1]
    '''
    img_np = np.float32(img_np) / 255.0 if img_np.dtype == np.uint8 else img_np
    degree, angle = int(degree), int(angle)
    if degree == 0:
        return img_np
    
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
 
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(img_np, -1, motion_blur_kernel)
 
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    return blurred / 255.0