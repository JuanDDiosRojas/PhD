import numpy as np
# import matplotlib.pyplot as plt
import h5py
# --- Dependencias de lenstronomy y astropy ---
from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import FlatLambdaCDM

from lenstronomy.Util.param_util import phi_q2_ellipticity, shear_polar2cartesian
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.image_util as image_util

# --- Parámetros cosmológicos y distancias ---
z_lens, z_source = 0.881, 2.059
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048)
D_d  = cosmo.angular_diameter_distance(z_lens)
D_s  = cosmo.angular_diameter_distance(z_source)
D_ds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)

# --- PixelGrid ---
deltaPix = 0.05
ra_at_xy_0, dec_at_xy_0 = -2.5, -2.5
transform_pix2angle = np.eye(2) * deltaPix
kwargs_pixel = {
    'nx': 100, 'ny': 100,
    'ra_at_xy_0': ra_at_xy_0,
    'dec_at_xy_0': dec_at_xy_0,
    'transform_pix2angle': transform_pix2angle
}
pixel_grid = PixelGrid(**kwargs_pixel)
x_coords, y_coords = pixel_grid.pixel_coordinates

# --- PSF ---
# Definición de la PSF (Point Spread Function)
# Aquí se define una PSF gaussiana con un FWHM de 0.1 arcsec
kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': 0.1, 'pixel_size': deltaPix}
psf = PSF(**kwargs_psf)


def simulate_forward(mass_subhalo, position_subhalo, lens_light=False):
    """
    Simulates strong gravitational lensing with an optional subhalo and lens light.

    Parameters
    ----------
    mass_subhalo : float
        Mass of the subhalo in solar masses (M_sun). Should be a positive float.
    position_subhalo : tuple or list of float
        Position of the subhalo in arcseconds, given as (x, y). Should be a tuple or list of two floats.
    lens_light : bool, optional
        If True, includes lens galaxy light in the simulation. Default is False.

    Returns
    -------
    tuple
        Returns (image_sub_noisy, image_nosub_clean, delta_psi_map, x_coords, y_coords)
    """
    
    # --- Modelos de luz ---
    lightModel_source = LightModel(light_model_list=['SERSIC'])
    kwargs_light_source = [{
        'amp': 100.0,
        'R_sersic': 0.15,
        'n_sersic': 1.0,
        'center_x': 0.0,
        'center_y': 0.0
    }]

    # --- Parámetros de la lente ---
    # Parámetros de la lente principal (Sersic)
    phi_lens, q_lens = np.deg2rad(-22.29), 0.866
    e1_lens, e2_lens = phi_q2_ellipticity(phi=phi_lens, q=q_lens)
    lightModel_lens = LightModel(light_model_list=['SERSIC_ELLIPSE'])
    if lens_light:
        amp_lens = 5.0  # Puedes ajustar este valor según lo que desees
    else:
        amp_lens = 0.0
    kwargs_light_lens = [{
        'amp': amp_lens,
        'R_sersic': 0.8,
        'n_sersic': 3.5,
        'e1': e1_lens,
        'e2': e2_lens,
        'center_x': 0.0,
        'center_y': 0.0
    }]

    # --- Modelos de lente ---
    # --- Parámetros de la lente principal (EPL: Elipsoidal Power Law) ---
    theta_E_main, gamma_main = 0.452, 2.042  # Radio de Einstein y pendiente de la ley de potencia
    e1_main, e2_main = phi_q2_ellipticity(phi=np.deg2rad(-22.29), q=0.866)  # Elipticidad de la lente principal
    kwargs_main = {
        'theta_E': theta_E_main, 'gamma': gamma_main,
        'e1': e1_main, 'e2': e2_main,
        'center_x': 0.0, 'center_y': -0.1  # Centro de la lente principal
    }

    # --- Parámetros del subhalo (SIS: Singular Isothermal Sphere) ---
    M_sub = mass_subhalo * u.M_sun  # Masa del subhalo
    # Cálculo del radio de Einstein del subhalo en radianes
    thetaE_sub_rad = np.sqrt(4*G*M_sub/c**2 * (D_ds/(D_d*D_s)))
    # Conversión del radio de Einstein a arcosegundos
    thetaE_sub = (thetaE_sub_rad * u.rad).to(u.arcsec).value
    kwargs_sub = {'theta_E': thetaE_sub, 'center_x': position_subhalo[0], 'center_y': position_subhalo[1]}  # Posición y radio de Einstein del subhalo

    # --- Parámetros de cizalla externa (SHEAR_REDUCED) ---
    g1, g2 = shear_polar2cartesian(phi=np.deg2rad(107.9), gamma=0.015)  # Conversión de cizalla polar a cartesiana
    kwargs_shear = {'gamma1': g1, 'gamma2': g2}

    # --- Lista de modelos de lente con subhalo ---
    lens_model_list = ['EPL', 'SIS', 'SHEAR_REDUCED']  # EPL: lente principal, SIS: subhalo, SHEAR_REDUCED: cizalla
    lensModel = LensModel(lens_model_list=lens_model_list)
    kwargs_lens = [kwargs_main, kwargs_sub, kwargs_shear]

    # --- Lista de modelos de lente sin subhalo ---
    lens_model_list_nosub = ['EPL', 'SHEAR_REDUCED']  # Sin el subhalo
    lensModel_nosub = LensModel(lens_model_list=lens_model_list_nosub)
    kwargs_lens_nosub = [kwargs_main, kwargs_shear]

    # --- Fuente puntual (quásar) ---
    # --- Definición del modelo de fuente puntual (quásar) ---
    # Lista de modelos de fuente puntual; 'SOURCE_POSITION' coloca la fuente puntual en una posición específica en el plano fuente
    point_source_model_list = ['SOURCE_POSITION']

    # Instancia del modelo de fuente puntual para el caso CON subhalo
    pointSource = PointSource(
        point_source_type_list=point_source_model_list,  # Tipos de fuentes puntuales
        lens_model=lensModel,                            # Modelo de lente (con subhalo)
        fixed_magnification_list=[True]                  # Mantener la magnificación fija
    )

    # Parámetros de la fuente puntual: posición (ra, dec) y amplitud
    kwargs_ps = [{
        'ra_source': 0.0,       # Posición en RA (arcsec)
        'dec_source': 0.0,      # Posición en Dec (arcsec)
        'source_amp': 100.0      # Amplitud de la fuente puntual
    }]

    # Instancia del modelo de fuente puntual para el caso SIN subhalo
    pointSource_nosub = PointSource(
        point_source_type_list=point_source_model_list,  # Tipos de fuentes puntuales
        lens_model=lensModel_nosub,                      # Modelo de lente (sin subhalo)
        fixed_magnification_list=[True]                  # Mantener la magnificación fija
    )

    # --- Parámetros numéricos para la simulación ---
    kwargs_numerics = {
        'supersampling_factor': 1,           # Factor de supermuestreo (1 = sin supermuestreo)
        'supersampling_convolution': False   # No aplicar convolución con supermuestreo
    }

    # --- ImageModel con subhalo ---
    imageModel = ImageModel(
        data_class=pixel_grid,
        psf_class=psf,
        lens_model_class=lensModel,
        source_model_class=lightModel_source,
        lens_light_model_class=lightModel_lens,
        point_source_class=pointSource,
        kwargs_numerics=kwargs_numerics
    )

    # --- ImageModel sin subhalo ---
    imageModel_nosub = ImageModel(
        data_class=pixel_grid,
        psf_class=psf,
        lens_model_class=lensModel_nosub,
        source_model_class=lightModel_source,
        lens_light_model_class=lightModel_lens,
        point_source_class=pointSource_nosub,
        kwargs_numerics=kwargs_numerics
    )

    # --- Simulación de imágenes ---
    exp_time = 100  # Tiempo de exposición en segundos
    background_rms = 0.1  # RMS del fondo en unidades de cuenta

    # a) Imagen CON subhalo y CON ruido
    image_sub = imageModel.image(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_light_source,
        kwargs_lens_light=kwargs_light_lens,
        kwargs_ps=kwargs_ps
    )
    poisson_sub = image_util.add_poisson(image_sub, exp_time=exp_time)
    bkg_sub = image_util.add_background(image_sub, sigma_bkd=background_rms)
    image_sub_noisy = image_sub + poisson_sub + bkg_sub

    # b) Imagen SIN subhalo y SIN ruido
    image_nosub_clean = imageModel_nosub.image(
        kwargs_lens=kwargs_lens_nosub,
        kwargs_source=kwargs_light_source,
        kwargs_lens_light=kwargs_light_lens,
        kwargs_ps=kwargs_ps
    )

    #-------- Potential correction for the noise in the image without subhalo
    # Aplanar las coordenadas para pasar al método de lenstronomy
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    # El método 'potential' devuelve el potencial en cada punto (x, y)
    psi_nosub = lensModel_nosub.potential(x_flat, y_flat, kwargs_lens_nosub)
    psi_nosub_map = psi_nosub.reshape(x_coords.shape)
    # Calcular el potencial lenteado para el modelo CON subhalo
    psi_sub = lensModel.potential(x_flat, y_flat, kwargs_lens)
    psi_sub_map = psi_sub.reshape(x_coords.shape)

    # Calcular la diferencia (corrección) entre ambos potenciales
    delta_psi_map = psi_sub_map - psi_nosub_map

    epsilon = 1e-3
    image_noisy_log = np.log10(image_sub_noisy + epsilon)
    image_clean_log = np.log10(image_nosub_clean + epsilon)
    delta_psi_log   = np.log10(np.abs(delta_psi_map) + epsilon) * np.sign(delta_psi_map)
    return image_noisy_log, image_clean_log, delta_psi_log

###############################################################################################
# --- Generación del dataset de imágenes simuladas ---
# Number of samples to generate
N = 300

# Ranges for subhalo mass and position
mass_min, mass_max = 1e7, 1e9       # [M_sun]
pos_min, pos_max   = -2.5, 2.5      # [arcsec]

# Pre–compute one example to grab array shape
_example_img, _example_smooth, _example_delta = simulate_forward(
    mass_subhalo=1e8,
    position_subhalo=(0.0, 0.0),
    lens_light=True
)
ny, nx = _example_img.shape

# Create HDF5 file and datasets
with h5py.File('lens_dataset.h5', 'w') as f:
    dset_input       = f.create_dataset('images_noisy',    (N, ny, nx), dtype='f4')
    dset_smooth      = f.create_dataset('images_smooth',   (N, ny, nx), dtype='f4')
    dset_delta_psi   = f.create_dataset('delta_psi_maps',  (N, ny, nx), dtype='f4')
    dset_mass        = f.create_dataset('subhalo_mass',    (N,),       dtype='f4')
    dset_xpos        = f.create_dataset('subhalo_x',       (N,),       dtype='f4')
    dset_ypos        = f.create_dataset('subhalo_y',       (N,),       dtype='f4')

    for i in range(N):
        # Sample mass log-uniformly
        logM = np.random.uniform(np.log10(mass_min), np.log10(mass_max))
        M_sub = 10**logM

        # Sample position uniformly
        x_sub = np.random.uniform(pos_min, pos_max)
        y_sub = np.random.uniform(pos_min, pos_max)

        # Run the forward simulation
        img_noisy, img_smooth, delta_psi = simulate_forward(
            mass_subhalo     = M_sub,
            position_subhalo = (x_sub, y_sub),
            lens_light       = True
        )

        # Store into HDF5
        dset_input[i, :, :]     = img_noisy
        dset_smooth[i, :, :]    = img_smooth
        dset_delta_psi[i, :, :] = delta_psi
        dset_mass[i]            = M_sub
        dset_xpos[i]            = x_sub
        dset_ypos[i]            = y_sub

    # Optionally add global attributes
    f.attrs['N_samples']     = N
    f.attrs['mass_range']    = (mass_min, mass_max)
    f.attrs['pos_range']     = (pos_min, pos_max)
    f.attrs['description']   = ("Dataset of simulated lensed images: "
                                "columns are noisy image, smooth model, and "
                                "potential correction δψ; accompanying subhalo "
                                "mass and (x,y) position for each sample.")


############################################################################################

# Vamos a generar un subplot con los tres mapas de imagenes:
# Genera un subplot con los tres mapas de imagen de la función simulate_forward
#### descomentar en caso de querer plotear los tres mapas de imagenes########
# 1) Define parámetros de ejemplo para el subhalo
# mass_subhalo     = 1e8          # [M_sun]
# position_subhalo = (0.5, 0.5)   # [arcsec]

# # 2) Ejecuta la simulación
# image_sub_noisy, image_nosub_clean, delta_psi_map= simulate_forward(
#     mass_subhalo,
#     position_subhalo
# )

# # 3) Visualiza los resultados en un subplot
# fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
# extent = [
#     x_coords.min(), x_coords.max(),
#     y_coords.min(), y_coords.max()
# ]

# # a) Imagen con subhalo + ruido
# im0 = axes[0].imshow(
#     image_sub_noisy, origin='lower',
#     cmap='gist_heat', extent=extent
# )
# axes[0].set_title('Con subhalo + ruido')
# axes[0].set_xlabel('RA offset [arcsec]')
# axes[0].set_ylabel('DEC offset [arcsec]')
# cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.85, pad=0.03)
# cbar0.set_label('log$_{10}$(counts flux)')

# # b) Imagen sin subhalo, sin ruido
# im1 = axes[1].imshow(
#     image_nosub_clean, origin='lower',
#     cmap='gist_heat', extent=extent
# )
# axes[1].set_title('Sin subhalo, sin ruido')
# axes[1].set_xlabel('RA offset [arcsec]')
# cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.85, pad=0.03)
# cbar1.set_label('log$_{10}$(counts flux)')

# # c) Corrección potencial Δψ
# im2 = axes[2].imshow(
#     delta_psi_map, origin='lower',
#     cmap='inferno', extent=extent
# )
# axes[2].set_title('Corrección: Δψ = ψ_con − ψ_sin')
# axes[2].set_xlabel('RA offset [arcsec]')
# cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
# cbar2.set_label('Δψ [arcsec²]')

# plt.tight_layout()
# plt.savefig('dark_satelite_simulation_log.png', dpi=300)
# plt.show()


###############################################################################################
# # Example usage

# # 1) Define example subhalo parameters
# mass_subhalo     = 1e8          # [M_sun]
# position_subhalo = (0.5, 0.5)   # [arcsec]

# # 2) Run the forward simulation
# image_sub_noisy, image_nosub_clean, delta_psi_map, x_coords, y_coords = simulate_forward(
#     mass_subhalo,
#     position_subhalo
# )

# # 3) Visualize side by side with matched WCS extents and individual colorbars
# fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
# extent = [
#     x_coords.min(), x_coords.max(),
#     y_coords.min(), y_coords.max()
# ]

# # a) Image with subhalo + noise
# im0 = axes[0].imshow(
#     np.log10(image_sub_noisy), origin='lower',
#     cmap='gist_heat', extent=extent
# )
# axes[0].set_title('Con subhalo + ruido')
# axes[0].set_xlabel('RA offset [arcsec]')
# axes[0].set_ylabel('DEC offset [arcsec]')
# cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.85, pad=0.03)
# cbar0.set_label('log$_{10}$(counts flux)')

# # b) Clean image without subhalo
# im1 = axes[1].imshow(
#     np.log10(image_nosub_clean), origin='lower',
#     cmap='gist_heat', extent=extent
# )
# axes[1].set_title('Sin subhalo, sin ruido')
# axes[1].set_xlabel('RA offset [arcsec]')
# cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.85, pad=0.03)
# cbar1.set_label('log$_{10}$(counts flux)')

# # c) Potential correction Δψ
# im2 = axes[2].imshow(
#     delta_psi_map, origin='lower',
#     cmap='inferno', extent=extent
# )
# axes[2].set_title('Corrección: Δψ = ψ_con − ψ_sin')
# axes[2].set_xlabel('RA offset [arcsec]')
# cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
# cbar2.set_label('Δψ [arcsec²]')

# plt.tight_layout()
# plt.savefig('dark_satelite_simulation.png', dpi=300)
# plt.show()
