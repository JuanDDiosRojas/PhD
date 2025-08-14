# import numpy as np
import h5py
# from astropy import units as u
# from astropy.constants import G, c
# from astropy.cosmology import FlatLambdaCDM
# from lenstronomy.Util.param_util import phi_q2_ellipticity, shear_polar2cartesian
# from lenstronomy.Data.pixel_grid import PixelGrid
# from lenstronomy.SimulationAPI.sim_api import SimAPI
# from lenstronomy.Survey.sim_surveys import LSST
# from lenstronomy.Util import util, plot_util
# from lenstronomy.LensModel.lens_model import LensModel
# import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import FlatLambdaCDM
import numpy as np

from lenstronomy.Util.param_util import phi_q2_ellipticity, shear_polar2cartesian
from lenstronomy.SimulationAPI.sim_api import SimAPI
import matplotlib.pyplot as plt
import lenstronomy.Plots.plot_util as plot_util
from lenstronomy.Data.pixel_grid import PixelGrid

# --- Parámetros globales ---
z_lens, z_source = 0.881, 2.059
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048)
D_d = cosmo.angular_diameter_distance(z_lens)
D_s = cosmo.angular_diameter_distance(z_source)
D_ds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)

# --- Configuración del grid de píxeles ---
numpix = 40
size = 6.0  # arcsec
pixel_scale = size / numpix
ra_at_xy_0, dec_at_xy_0 = -size / 2, -size / 2
transform_pix2angle = np.eye(2) * pixel_scale

kwargs_pixel = {
    'nx': numpix,
    'ny': numpix,
    'ra_at_xy_0': ra_at_xy_0,
    'dec_at_xy_0': dec_at_xy_0,
    'transform_pix2angle': transform_pix2angle
}

pixel_grid = PixelGrid(**kwargs_pixel)
x_coords, y_coords = pixel_grid.pixel_coordinates
x_flat, y_flat = x_coords.flatten(), y_coords.flatten()

# --- Configuración de bandas LSST ---
LSST_g = LSST(band='g', psf_type='GAUSSIAN', coadd_years=10)
LSST_r = LSST(band='r', psf_type='GAUSSIAN', coadd_years=10)
LSST_i = LSST(band='i', psf_type='GAUSSIAN', coadd_years=10)
lsst_bands = [LSST_g, LSST_r, LSST_i]

# --- Parámetros de magnitudes por banda ---
mag_g = 22.0
g_r = 1.0
g_i = 2.0
R_sersic = 0.15
n_sersic = 1.0

kwargs_source_mag_g = [{'magnitude': mag_g, 'R_sersic': R_sersic, 'n_sersic': n_sersic,
                        'center_x': 0.0, 'center_y': 0.0}]
kwargs_source_mag_r = [{'magnitude': mag_g - g_r, **kwargs_source_mag_g[0]}]
kwargs_source_mag_i = [{'magnitude': mag_g - g_i, **kwargs_source_mag_g[0]}]
source_mags = [kwargs_source_mag_g, kwargs_source_mag_r, kwargs_source_mag_i]

# --- Parámetros del modelo de lente principal ---
theta_E_main, gamma_main = 1.452, 1.9
e1_main, e2_main = phi_q2_ellipticity(phi=np.deg2rad(-22.29), q=0.866)
kwargs_main = {
    'theta_E': theta_E_main, 'gamma': gamma_main,
    'e1': e1_main, 'e2': e2_main,
    'center_x': 0.0, 'center_y': -0.1
}
g1, g2 = shear_polar2cartesian(phi=np.deg2rad(107.9), gamma=0.015)
kwargs_shear = {'gamma1': g1, 'gamma2': g2}

# --- Escalado de imagen RGB ---
def scale_image_rgb(img):
    def _scale_max(image):
        flat = image.flatten()
        flat.sort()
        return flat[int(len(flat)*0.95)]
    img_scaled = np.zeros_like(img)
    for i in range(3):
        img_scaled[..., i] = plot_util.sqrt(img[..., i], scale_min=0, scale_max=_scale_max(img[..., i]))
    return img_scaled

# --- Simulación RGB ---
def simulate_rgb_image(band_configs, model_config, kwargs_lens, kwargs_source_mags, size=6.0, kwargs_numerics=None, add_noise=True):
    images = []
    for band_config, kwargs_source_mag in zip(band_configs, kwargs_source_mags):
        kw_band = band_config.kwargs_single_band()
        pixel_scale = 0.15
        numpix = int(round(size / pixel_scale))
        sim = SimAPI(numpix=numpix, kwargs_single_band=kw_band, kwargs_model=model_config)
        _, kwargs_source, _ = sim.magnitude2amplitude([], kwargs_source_mag, [])
        imSim = sim.image_model_class(kwargs_numerics)
        image = imSim.image(kwargs_lens, kwargs_source, None, None)
        if add_noise:
            image += sim.noise_for_model(model=image)
        images.append(image)
    return np.stack(images, axis=-1)

# --- Función principal para generar una muestra ---
def simulate_pair_and_delta(mass_subhalo, position_subhalo):
    M_sub = mass_subhalo * u.M_sun
    thetaE_sub_rad = np.sqrt(4*G*M_sub/c**2 * (D_ds/(D_d*D_s)))
    thetaE_sub = (thetaE_sub_rad * u.rad).to(u.arcsec).value
    kwargs_sub = {'theta_E': thetaE_sub, 'center_x': position_subhalo[0], 'center_y': position_subhalo[1]}

    kwargs_lens = [kwargs_main, kwargs_sub, kwargs_shear]
    kwargs_lens_nosub = [kwargs_main, kwargs_shear]

    kwargs_model_sub = {'lens_model_list': ['EPL', 'SIS', 'SHEAR_REDUCED'], 'source_light_model_list': ['SERSIC'], 'lens_light_model_list': [], 'point_source_model_list': []}
    kwargs_model_nosub = {'lens_model_list': ['EPL', 'SHEAR_REDUCED'], 'source_light_model_list': ['SERSIC'], 'lens_light_model_list': [], 'point_source_model_list': []}

    kwargs_numerics = {'point_source_supersampling_factor': 1}

    img_clean = simulate_rgb_image(lsst_bands, kwargs_model_nosub, kwargs_lens_nosub, source_mags, size=6.0, kwargs_numerics=kwargs_numerics, add_noise=False)
    img_subhalo = simulate_rgb_image(lsst_bands, kwargs_model_sub, kwargs_lens, source_mags, size=6.0, kwargs_numerics=kwargs_numerics, add_noise=True)

    lensModel = LensModel(lens_model_list=kwargs_model_sub['lens_model_list'])
    lensModel_nosub = LensModel(lens_model_list=kwargs_model_nosub['lens_model_list'])
    psi_sub = lensModel.potential(x_flat, y_flat, kwargs_lens).reshape(x_coords.shape)
    psi_nosub = lensModel_nosub.potential(x_flat, y_flat, kwargs_lens_nosub).reshape(x_coords.shape)
    delta_psi_map = psi_sub - psi_nosub
    epsilon = 1e-3
    delta_psi_log = np.log10(np.abs(delta_psi_map) + epsilon) * np.sign(delta_psi_map)

    return scale_image_rgb(img_subhalo), scale_image_rgb(img_clean), delta_psi_log


#----------Loop to generate dataset-------------------
# --- Configuración ---
N = 5  # Número de muestras
mass_min, mass_max = 1e7, 1e9  # Masa del subhalo [M_sun]
pos_min, pos_max   = -1.6, 1.6  # Rango de posiciones (x,y) en arcsec

# --- Determinar forma de salida RGB ---
_example_rgb, _example_clean, _example_delta = simulate_pair_and_delta(
    mass_subhalo=1e8,
    position_subhalo=(0.0, 0.5)
)
ny, nx, n_channels = _example_rgb.shape  # asume RGB = 3 canales

# --- Crear archivo HDF5 ---
with h5py.File('lens_dataset_lsst_rgb.h5', 'w') as f:
    dset_input       = f.create_dataset('images_rgb',      (N, ny, nx, n_channels), dtype='f4')
    dset_clean       = f.create_dataset('images_clean',    (N, ny, nx, n_channels), dtype='f4')
    dset_delta_psi   = f.create_dataset('delta_psi_maps',  (N, ny, nx),             dtype='f4')
    dset_mass        = f.create_dataset('subhalo_mass',    (N,),                    dtype='f4')
    dset_xpos        = f.create_dataset('subhalo_x',       (N,),                    dtype='f4')
    dset_ypos        = f.create_dataset('subhalo_y',       (N,),                    dtype='f4')

    for i in range(N):
        # Muestreo de parámetros
        logM = np.random.uniform(np.log10(mass_min), np.log10(mass_max))
        M_sub = 10**logM
        x_sub = np.random.uniform(pos_min, pos_max)
        y_sub = np.random.uniform(pos_min, pos_max)

        # Simulación
        img_rgb, img_clean, delta_psi = simulate_pair_and_delta(
            mass_subhalo     = M_sub,
            position_subhalo = (x_sub, y_sub)
        )

        # Almacenamiento
        dset_input[i]       = img_rgb
        dset_clean[i]       = img_clean
        dset_delta_psi[i]   = delta_psi
        dset_mass[i]        = M_sub
        dset_xpos[i]        = x_sub
        dset_ypos[i]        = y_sub

    # Metadatos opcionales
    f.attrs['N_samples']   = N
    f.attrs['mass_range']  = (mass_min, mass_max)
    f.attrs['pos_range']   = (pos_min, pos_max)
    f.attrs['description'] = ("Simulated RGB gravitational lensing dataset (LSST-like): "
                              "Each entry has noisy RGB image, clean RGB image, and delta_psi. "
                              "Labels include subhalo mass and (x,y) position.")

