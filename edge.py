import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import random
import cv2
import numpy as np


def verify_input_shape(image, function_name):
    if image.shape != (1, 256, 256):
        raise ValueError(f"{function_name}: Input image must have shape [1, 256, 256], got {image.shape}")
    # Add batch dimension if needed for convolution operations
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # Shape becomes [1, 1, 256, 256]
    return image


def create_sobel_kernels():
    # Sobel kernels for x and y directions
    sobel_x = torch.tensor([[-1, 0, 1], 
                           [-2, 0, 2], 
                           [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], 
                           [0, 0, 0], 
                           [1, 2, 1]], dtype=torch.float32)
    
    return (sobel_x.view(1, 1, 3, 3), 
            sobel_y.view(1, 1, 3, 3))


def sobel(image):
    # Verify input shape and add batch dimension
    image = verify_input_shape(image, "Sobel")

    # Get Sobel kernels
    sobel_x, sobel_y = create_sobel_kernels()
    sobel_x = sobel_x.to(image.device)
    sobel_y = sobel_y.to(image.device)
    
    # Apply Sobel filters
    gx = F.conv2d(image, sobel_x, padding=1)
    gy = F.conv2d(image, sobel_y, padding=1)
    
    # Calculate gradient magnitude
    edge = torch.sqrt(gx**2 + gy**2)
    
    # Normalize
    edge = (edge - edge.min()) / (edge.max() - edge.min())
    return edge.squeeze(0)


def tensor_to_opencv(image_tensor):
    image_np = image_tensor.cpu().numpy()
    if len(image_np.shape) == 4:
        image_np = image_np[0, 0]
    elif len(image_np.shape) == 3:
        image_np = image_np[0]
    image_np = (image_np * 255).astype(np.uint8)
    return image_np


def canny(image, low_thr=25, high_thr=230):
    image_np = tensor_to_opencv(image)
    edges = cv2.Canny(image_np, low_thr, high_thr)
    edges = torch.from_numpy(edges).float() / 255.0
    return edges.unsqueeze(0)


def create_gaussian_kernel(kernel_size, sigma):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    # make a grid of (x, y) coordinates
    ax = torch.linspace(-(kernel_size // 2), kernel_size // 2, steps=kernel_size)
    xx, yy = torch.meshgrid([ax, ax], indexing='ij')
    xx = xx.float()
    yy = yy.float()
    # compute the 2D Gaussian function
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    # normalize the kernel so that sum is 1
    kernel = kernel / torch.sum(kernel)
    # check the shape before reshaping
    if kernel.numel() != kernel_size * kernel_size:
        raise RuntimeError(f"Kernel size mismatch: expected {kernel_size * kernel_size}, got {kernel.numel()}")
    # reshape to be compatible with conv2d (out_channels, in_channels, H, W)
    return kernel.view(1, 1, kernel_size, kernel_size)


def create_laplacian_kernel():
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    #laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
    return laplacian_kernel.view(1, 1, 3, 3)


def abslog(image, kernel_size=3, sigma=0.6):
    # create the Gaussian and Laplacian kernels
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma).to(image.device)
    laplacian_kernel = create_laplacian_kernel().to(image.device)
    # apply the Gaussian smoothing
    smoothed = F.conv2d(image, gaussian_kernel, padding=kernel_size // 2)
    # apply the Laplacian filter
    log_out = F.conv2d(smoothed, laplacian_kernel, padding=1)
    # abs and norm
    log_out = torch.abs(log_out)
    edge = (log_out - log_out.min()) / (log_out.max() - log_out.min())
    return edge


def anisotropic_diffusion(img, n_iter=10, kappa=25, gamma=0.1):
    if len(img.shape) != 3 or img.shape[0] != 1:
        raise ValueError("Input should be a grayscale image tensor of shape [1, H, W]")
    
    # Padding the input to handle boundaries
    padded = F.pad(img, (1, 1, 1, 1), mode='reflect')
    
    # Initialize output
    diff = torch.zeros_like(img)
    
    for _ in range(n_iter):
        # Calculate gradients in all directions
        north = padded[:, :-2, 1:-1] - padded[:, 1:-1, 1:-1]
        south = padded[:, 2:, 1:-1] - padded[:, 1:-1, 1:-1]
        east = padded[:, 1:-1, 2:] - padded[:, 1:-1, 1:-1]
        west = padded[:, 1:-1, :-2] - padded[:, 1:-1, 1:-1]
        
        # Calculate conduction coefficients
        c_north = torch.exp(-(north/kappa)**2)
        c_south = torch.exp(-(south/kappa)**2)
        c_east = torch.exp(-(east/kappa)**2)
        c_west = torch.exp(-(west/kappa)**2)
        
        # Update image
        diff = gamma * (c_north*north + c_south*south + c_east*east + c_west*west)
        padded = F.pad(padded[:, 1:-1, 1:-1] + diff, (1, 1, 1, 1), mode='reflect')

    return padded[:, 1:-1, 1:-1]


def two_abslog(img, kernel_size=3, sigma=0.6):
    return abslog(abslog(img, kernel_size, sigma), kernel_size, sigma)


def apply_effects(edge_map):
    # Assume edge_map has shape [1, H, W]
    _, H, W = edge_map.shape
    
    # 1. Random thinner or disperse edge
    if random.random() < 0.5:
        # Generate random mask (30% of pixels selected)
        
        mask_prob = np.random.uniform(0.5 * 0.4, 1.5 * 0.4)
        region_mask = (torch.rand((1, H, W), device=edge_map.device) < mask_prob).float()

        choice = random.choice(['random_kernel', 'gaussian_blur', 'morphological'])
        
        if choice == 'random_kernel':
            # Random convolution kernel for thinning
            kernel_vals = torch.rand((3, 3))
            kernel_vals[1, 1] = 0.0  # center weight = 0
            kernel = kernel_vals / kernel_vals.sum()
            kernel = kernel.unsqueeze(0).unsqueeze(0).to(edge_map.device)
            edge_map = F.conv2d(edge_map.unsqueeze(0), kernel, padding=1).squeeze(0)
        
        elif choice == 'gaussian_blur':
            # Apply a soft blur to simulate dispersion
            sigma = random.uniform(0.5, 1.5)
            x = torch.arange(-1, 2).float()
            gauss = torch.exp(-x**2 / (2 * sigma**2))
            gauss_kernel = (gauss[:, None] * gauss[None, :])
            gauss_kernel /= gauss_kernel.sum()
            kernel = gauss_kernel.unsqueeze(0).unsqueeze(0).to(edge_map.device)
            edge_map = F.conv2d(edge_map.unsqueeze(0), kernel, padding=1).squeeze(0)

        elif choice == 'morphological':
            # Use max-pooling then subtract to "thin"
            pooled = F.max_pool2d(edge_map.unsqueeze(0), 3, stride=1, padding=1)
            edge_map = torch.clamp(edge_map - pooled.squeeze(0) * 0.3, 0.0, 1.0)

    # 3. Random contrast modification
    if random.random() < 0.5:
        factor = random.uniform(0.5, 1.3)
        edge_map = torch.pow(edge_map, factor)

    edge_map = (edge_map - edge_map.min())/(edge_map.max() - edge_map.min())

    return edge_map


def generate_random_half_mask(image_tensor):
    _, H, W = image_tensor.shape
    center_x, center_y = W // 2, H // 2

    # Create coordinate grid
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    x = x - center_x
    y = y - center_y

    # Random angle between 0 and 2π
    theta = torch.rand(1).item() * 2 * np.pi

    # Line normal vector
    n_x = math.cos(theta)
    n_y = math.sin(theta)

    # Compute dot product between normal and each (x,y) point
    dot = x * n_x + y * n_y

    # Boolean mask
    mask = (dot >= 0)  # dtype=torch.bool

    return mask.unsqueeze(0)


def extract_edge_func(img, if_aug=True):
    if not if_aug:
        img  = anisotropic_diffusion(img)
        return sobel(img)

    # first run anisotropic diffusion to smooth the image
    kappa = np.random.uniform(0.8 * 25, 1.2 * 25)
    gamma = np.random.uniform(0.5 * 0.1, 2.0 * 0.1)
    n_iter = 10
    img = anisotropic_diffusion(img, n_iter, kappa, gamma)

    # generate random parameters
    kernel_size = 3
    sigma = np.random.uniform(0.5 * 0.6, 1.5 * 0.6)
    low_thr = int(np.random.uniform(15, 60))
    high_thr = int(np.random.uniform(200, 245))
    
    # random method list
    methods_list = [
        lambda x: abslog(x, kernel_size, sigma),
        lambda x: two_abslog(x, kernel_size, sigma),
        lambda x: sobel(x),
        lambda x: canny(x, low_thr, high_thr)
    ]
    
    # select one or two methods
    rand_val = random.random()
    if rand_val < 1/2:
        method = random.choice(methods_list)
        init_edge = method(img)
    else:
        selected_methods = random.sample(methods_list, 2)
        result1 = selected_methods[0](img)
        result2 = selected_methods[1](img)
        ratio = random.random()
        init_edge = ratio * result1 + (1 - ratio) * result2
        init_edge = (init_edge - init_edge.min()) / (init_edge.max() - init_edge.min())

    # randomly apply effects
    rand_val = random.random()
    if rand_val < 1/2:
        edge_map = apply_effects(init_edge)
    else:
        rand_map1 = apply_effects(init_edge)
        rand_map2 = apply_effects(init_edge)
        mask = generate_random_half_mask(init_edge)
        edge_map = torch.where(mask, rand_map1, rand_map2)
    
    edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())

    return edge_map


def extract_edge(img, if_aug=True):
    is_nan = True
    count = 0
    while is_nan:
        edge_map = extract_edge_func(img, if_aug)
        is_nan = torch.isnan(edge_map).any()
        count += 1
        if count > 100:
            edge_map = torch.rand(img.shape) + 1e-6
            break
    return edge_map


if __name__ == "__main__":
    for _ in range(100):
        img = torch.randn(1, 256, 256)
        edge_map = extract_edge(img)
    edge_map = extract_edge(img)
    print(f"Edge map shape: {edge_map.shape}")
