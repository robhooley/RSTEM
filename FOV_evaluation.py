import cv2
import matplotlib.pyplot as plt
import numpy as np

from expert_pi.RSTEM import fft_processing, fov_fine_tuning


def get_ellipse_parameters(scale_x, scale_y, shear):
    """Function to transform anisotropic scale and shear to parameters of ellipse parameters.

    :param scale_x: scale in x direction
    :param scale_y: scale in y direction
    :param shear: horizontal shear
    :return: R1, R2, theta - main semi-axes of the ellipse and its rotation angle in degrees
    """
    # Sestavení transformační matice
    T = np.array([[scale_x, shear], [0, scale_y]])

    # Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(T)

    # Hlavní poloosy elipsy (R1, R2) jsou singular values
    R1, R2 = S

    # Úhel natočení elipsy je dán prvním sloupcem U
    theta = np.arctan2(U[1, 0], U[0, 0])

    return R1, R2, np.degrees(theta)


def get_distortion_parameters(scale_x, scale_y, shear):
    """Function to transform anisotropic scale and shear to parameters of elliptical distortion.

    :param scale_x: scale in x direction
    :param scale_y: scale in y direction
    :param shear: horizontal shear
    :return: error of fov calibration, image distortion (E1-E2), parameters of eliptical distortion (E1, E2, theta)
    """
    R1, R2, theta = get_ellipse_parameters(scale_x, scale_y, shear)
    E1 = R1 - 1
    E2 = R2 - 1
    image_distortion = E1 - E2
    calibration_error = np.sqrt(np.abs(E1 * E2))

    return calibration_error, image_distortion, (E1, E2, theta)


def calculate_image_distortion(
    image_path,
    fov,
    peak_positions,
    peak_area_size=15,
    reflections=["111"],
    max_fov_error=0.05,
    plot=False,
    scan_rotation_deg=0
):
    """
    Calculate parameters of elliptical distortion from a HR silicon image.

    :param image_path: path to high-resolution silicon image.
    :param fov: Field of view in nanometers.
    :param peak_positions: Initial peak positions for FFT refinement. List of (x, y) tuples in pixel coordinates.
    :param peak_area_size: Size of the area around each peak to consider for refinement.
    :param max_fov_error: Maximum allowed field of view error. (used for clustering peaks)
    :param reflections: List of reflections to consider, either ["111"] or ["111", "200"].
    :param plot: Whether to plot the results.
    :param scan_rotation_deg: Acquisition rotation used to acquire the scan image in degrees.
    :return: error of fov calibration, image distortion (E1-E2), parameters of eliptical distortion (E1, E2, theta)
    """
    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

    fft_peaks = fft_processing.refine_fft_peak_positions(image, peak_positions, peak_area_size, plot=plot)
    peak_clusters = fft_processing.filter_fft_peaks(fft_peaks, fov=fov, max_fov_error=max_fov_error, min_intensity=None, fft_shape=image.shape)

    # Generate expected points
    expected_points = fov_fine_tuning.generate_Si_reflections(fov, reflections=reflections, center=(image.shape[1] // 2, image.shape[0] // 2), include_center=True)

    # Sort measured points
    measured_points = fov_fine_tuning.sort_measured_points(peak_clusters, reflections=reflections, include_center=True)

    # Calculate scales and shear
    scale_x, scale_y, shear = fov_fine_tuning.get_scales_shear(expected_points, measured_points, output=plot, rot_deg=scan_rotation_deg)

    calibration_error, image_distortion, (E1, E2, theta) = get_distortion_parameters(scale_x, scale_y, shear)

    return calibration_error, image_distortion, (E1, E2, theta)


image_path = r"C:\Users\robert.hooley\Documents\Acceptance\P6\Rob acceptance testing\STEM0002.tiff"
fov=35

img = cv2.imread(image_path,cv2.IMREAD_ANYDEPTH)

fft_result = np.fft.fft2(img)
fft_shifted = np.fft.fftshift(fft_result)
magnitude = np.abs(fft_shifted)
log_magnitude = np.log1p(magnitude)

plt.figure(figsize=(10, 10))  # defines plot size
plt.gray()  # sets it to grayscale
plt.title("Click to place FFT spots, right click to remove and middle click to finish, max masks 8")
plt.imshow(log_magnitude)  # plots the representative diffraction and scales the intensity
peak_positions = plt.ginput(n=8, show_clicks=True, timeout=0)  # user interacts to define mask positions
plt.close()  # closes the plot


results = calculate_image_distortion(    image_path,    fov,   peak_positions,    peak_area_size=15,    reflections=["111","200"],    max_fov_error=0.1,    plot=True,    scan_rotation_deg=0)
for i in results:
    print(i)