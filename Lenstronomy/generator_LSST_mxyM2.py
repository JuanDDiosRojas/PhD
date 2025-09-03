# -*- coding: utf-8 -*-
# Generador de dataset LSST con subhalos para VAE/Regresión
# Correcciones: grid consistente, magnitudes por banda correctas, Δψ consistente, masa log-uniforme.

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import h5py
from lenstronomy.Util.param_util import phi_q2_ellipticity, shear_polar2cartesian
from lenstronomy.SimulationAPI.sim_api import SimAPI
import lenstronomy.Plots.plot_util as plot_util
from lenstronomy.Data.pixel_grid import PixelGrid  # asegúrate de tener esta import
# -----------------------------
# Configuración global
# -----------------------------
SEED = 12345
rng = np.random.default_rng(SEED)

# Cosmología y distancias
z_lens, z_source = 0.881, 2.059
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048)
D_d  = cosmo.angular_diameter_distance(z_lens)
D_s  = cosmo.angular_diameter_distance(z_source)
D_ds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)

# Bandas LSST
LSST_g = LSST(band='g', psf_type='GAUSSIAN', coadd_years=10)
LSST_r = LSST(band='r', psf_type='GAUSSIAN', coadd_years=10)
LSST_i = LSST(band='i', psf_type='GAUSSIAN', coadd_years=10)
lsst_bands = [LSST_g, LSST_r, LSST_i]

# Parámetros fotométricos de la fuente
mag_g = 22.0
g_r = 1.0   # g - r
g_i = 2.0   # g - i
R_sersic = 0.15
n_sersic = 1.0

# Lente principal (EPL) y cizalla
theta_E_main, gamma_main = 1.452, 1.9
e1_main, e2_main = phi_q2_ellipticity(phi=np.deg2rad(-22.29), q=0.866)
kwargs_main = dict(theta_E=theta_E_main, gamma=gamma_main,
                   e1=e1_main, e2=e2_main, center_x=0.0, center_y=-0.1)
g1, g2 = shear_polar2cartesian(phi=np.deg2rad(107.9), gamma=0.015)
kwargs_shear = dict(gamma1=g1, gamma2=g2)

# Parámetros de imagen
STAMP_SIZE_ARCSEC = 6.0     # tamaño total (lado) del recorte, en arcsec
PIXEL_SCALE = 0.15          # arcsec/pixel (-> 40x40)
NUMPIX = int(round(STAMP_SIZE_ARCSEC / PIXEL_SCALE))
KWARGS_NUMERICS = {'point_source_supersampling_factor': 1}

# ¿Guardar imágenes con sqrt-stretch (true) o lineales (false)?
DO_SCALE = True


# -----------------------------
# Utilidades
# -----------------------------
def scale_pair_rgb(img_noisy, img_clean):
    """Aplica sqrt-stretch por canal usando el percentil 95 de la imagen clean;
    devuelve (noisy_scaled, clean_scaled, v_por_banda)."""
    def _scale_max(image):
        flat = image.flatten()
        flat.sort()
        return flat[int(len(flat) * 0.95)]
    v = np.array([_scale_max(img_clean[..., i]) for i in range(img_clean.shape[-1])], dtype=float)
    out_noisy = np.zeros_like(img_noisy, dtype=float)
    out_clean = np.zeros_like(img_clean, dtype=float)
    for i in range(img_clean.shape[-1]):
        out_noisy[..., i] = plot_util.sqrt(img_noisy[..., i], scale_min=0, scale_max=v[i])
        out_clean[..., i] = plot_util.sqrt(img_clean[..., i], scale_min=0, scale_max=v[i])
    return out_noisy, out_clean, v


def simulate_rgb_image(band_configs, model_config, kwargs_lens, kwargs_source_mags,
                       add_noise=True):
    """
    Genera imagen multibanda y devuelve también el grid (x,y) del imSim.
    Robusto a diferencias de versión en lenstronomy.
    """
    images = []
    xy_grid = None
    for band_config, kwargs_source_mag in zip(band_configs, kwargs_source_mags):
        kw_band = band_config.kwargs_single_band()
        sim = SimAPI(numpix=NUMPIX, kwargs_single_band=kw_band, kwargs_model=model_config)
        _, kwargs_source, _ = sim.magnitude2amplitude([], kwargs_source_mag, [])
        imSim = sim.image_model_class(KWARGS_NUMERICS)

        image = imSim.image(kwargs_lens, kwargs_source, None, None)
        if add_noise:
            image += sim.noise_for_model(model=image)
        images.append(image)

        # --- obtener el grid del propio ImageModel (compatibilidad multi-versión)
        if xy_grid is None:
            # 1) Intenta Data.pixel_grid.pixel_coordinates (versiones nuevas)
            data_obj = getattr(imSim, 'Data', None) or getattr(imSim, 'data_class', None) or getattr(sim, 'data_class', None)
            got = False
            if data_obj is not None:
                try:
                    x_coords, y_coords = data_obj.pixel_grid.pixel_coordinates
                    got = True
                except Exception:
                    pass
                if not got and hasattr(data_obj, 'pixel_coordinates'):
                    try:
                        x_coords, y_coords = data_obj.pixel_coordinates
                        got = True
                    except Exception:
                        pass
                if not got and hasattr(data_obj, 'x_grid') and hasattr(data_obj, 'y_grid'):
                    # Algunos ImageData exponen x_grid/y_grid aplanados
                    try:
                        x_coords = np.asarray(data_obj.x_grid).reshape(NUMPIX, NUMPIX)
                        y_coords = np.asarray(data_obj.y_grid).reshape(NUMPIX, NUMPIX)
                        got = True
                    except Exception:
                        pass
                if not got and hasattr(data_obj, 'grid_class') and hasattr(data_obj.grid_class, 'pixel_coordinates'):
                    try:
                        x_coords, y_coords = data_obj.grid_class.pixel_coordinates
                        got = True
                    except Exception:
                        pass

            # 2) Fallback: recrea el grid con los mismos parámetros globales
            if not got:
                ra0 = -STAMP_SIZE_ARCSEC / 2.0
                dec0 = -STAMP_SIZE_ARCSEC / 2.0
                transform = np.eye(2) * PIXEL_SCALE
                pg = PixelGrid(nx=NUMPIX, ny=NUMPIX,
                               ra_at_xy_0=ra0, dec_at_xy_0=dec0,
                               transform_pix2angle=transform)
                x_coords, y_coords = pg.pixel_coordinates

            xy_grid = (x_coords, y_coords)

    return np.stack(images, axis=-1), xy_grid

# -----------------------------
# Núcleo: una muestra (imágenes + Δψ)
# -----------------------------
def simulate_pair_and_delta(mass_subhalo, position_subhalo, source_position,
                            images_use_reduced_shear=True, scale_outputs=DO_SCALE):
    """Devuelve (img_subhalo, img_clean, delta_psi_log)."""

    # Radio de Einstein del subhalo (SIS)
    M_sub = mass_subhalo * u.M_sun
    thetaE_sub_rad = np.sqrt(4 * G * M_sub / c**2 * (D_ds / (D_d * D_s)))
    thetaE_sub = (thetaE_sub_rad * u.rad).to(u.arcsec).value

    kwargs_sub = {'theta_E': thetaE_sub, 'center_x': position_subhalo[0], 'center_y': position_subhalo[1]}
    kwargs_lens       = [kwargs_main, kwargs_sub, kwargs_shear]
    kwargs_lens_nosub = [kwargs_main,           kwargs_shear]

    # Modelos para las IMÁGENES
    if images_use_reduced_shear:
        lens_list_img_sub   = ['EPL', 'SIS', 'SHEAR_REDUCED']
        lens_list_img_nosub = ['EPL',       'SHEAR_REDUCED']
    else:
        lens_list_img_sub   = ['EPL', 'SIS', 'SHEAR']
        lens_list_img_nosub = ['EPL',       'SHEAR']

    kwargs_model_sub_img = {'lens_model_list': lens_list_img_sub, 'source_light_model_list': ['SERSIC'],
                            'lens_light_model_list': [], 'point_source_model_list': []}
    kwargs_model_nosub_img = {'lens_model_list': lens_list_img_nosub, 'source_light_model_list': ['SERSIC'],
                              'lens_light_model_list': [], 'point_source_model_list': []}

    # Modelo para el POTENCIAL (siempre con SHEAR para que exista ψ)
    lens_list_psi_sub   = ['EPL', 'SIS', 'SHEAR']
    lens_list_psi_nosub = ['EPL',       'SHEAR']

    # Fuente (magnitudes por banda correctas + centro variable)
    x_src, y_src = source_position
    base = {'R_sersic': R_sersic, 'n_sersic': n_sersic, 'center_x': x_src, 'center_y': y_src}
    kwargs_source_mag_g = [{**base, 'magnitude': mag_g}]
    kwargs_source_mag_r = [{**base, 'magnitude': mag_g - g_r}]
    kwargs_source_mag_i = [{**base, 'magnitude': mag_g - g_i}]
    source_mags = [kwargs_source_mag_g, kwargs_source_mag_r, kwargs_source_mag_i]

    # Imágenes: clean (sin subhalo) y con subhalo (ruido activado)
    img_clean_lin, (x_coords, y_coords) = simulate_rgb_image(
        lsst_bands, kwargs_model_nosub_img, kwargs_lens_nosub, source_mags, add_noise=False
    )
    img_sub_lin, _ = simulate_rgb_image(
        lsst_bands, kwargs_model_sub_img, kwargs_lens, source_mags, add_noise=True
    )

    # Potencial en el MISMO grid
    x_flat, y_flat = x_coords.ravel(), y_coords.ravel()
    lensModel_psi       = LensModel(lens_model_list=lens_list_psi_sub)
    lensModel_nosub_psi = LensModel(lens_model_list=lens_list_psi_nosub)
    psi_sub   = lensModel_psi.potential(x_flat, y_flat, kwargs_lens).reshape(x_coords.shape)
    psi_nosub = lensModel_nosub_psi.potential(x_flat, y_flat, kwargs_lens_nosub).reshape(x_coords.shape)
    delta_psi_map = psi_sub - psi_nosub
    epsilon = 1e-3
    delta_psi_log = np.log10(np.abs(delta_psi_map) + epsilon) * np.sign(delta_psi_map)

    if scale_outputs:
        img_sub, img_clean, _ = scale_pair_rgb(img_sub_lin, img_clean_lin)
        return img_sub, img_clean, delta_psi_log
    else:
        return img_sub_lin, img_clean_lin, delta_psi_log


# -----------------------------
# Generación del dataset
# -----------------------------
if __name__ == "__main__":
    # Configuración dataset
    N = 6000
    mass_min, mass_max = 1e7, 1e9
    subhalo_pos_min, subhalo_pos_max = -1.6, 1.6  # subhalo en un área más pequeña
    source_pos_min,  source_pos_max  = -0.3, 0.3  # fuente casi centrada

    # Ejemplo para dimensiones
    ex_rgb, ex_clean, ex_dpsi = simulate_pair_and_delta(
        mass_subhalo=1e8,
        position_subhalo=(0.0, 0.5),
        source_position=(0.0, 0.0),
        images_use_reduced_shear=True,  # puedes poner False si quieres SHEAR también en imágenes
        scale_outputs=DO_SCALE
    )
    ny, nx, n_channels = ex_rgb.shape

    # Archivo HDF5
    with h5py.File('LSST_Mmxy2.h5', 'w') as f:
        # Datasets (chunked + compresión)
        dset_input     = f.create_dataset('images_rgb',     (N, ny, nx, n_channels), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_clean     = f.create_dataset('images_clean',   (N, ny, nx, n_channels), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_delta_psi = f.create_dataset('delta_psi_maps', (N, ny, nx),             dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_mass      = f.create_dataset('subhalo_mass',   (N,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_xpos      = f.create_dataset('subhalo_x',      (N,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_ypos      = f.create_dataset('subhalo_y',      (N,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_srcx      = f.create_dataset('source_x',       (N,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_srcy      = f.create_dataset('source_y',       (N,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)

        for i in range(N):
            # Masa log-uniforme
            # logM = rng.uniform(np.log10(mass_min), np.log10(mass_max))
            # M_sub = 10.0**logM
            M_sub = rng.uniform(mass_min, mass_max)

            # Posición del subhalo
            x_sub = rng.uniform(subhalo_pos_min, subhalo_pos_max)
            y_sub = rng.uniform(subhalo_pos_min, subhalo_pos_max)

            # Posición de la fuente (jitter uniforme; cambia a normal si prefieres)
            x_src = rng.uniform(source_pos_min, source_pos_max)
            y_src = rng.uniform(source_pos_min, source_pos_max)

            # Simulación
            img_rgb, img_clean, delta_psi = simulate_pair_and_delta(
                mass_subhalo     = M_sub,
                position_subhalo = (x_sub, y_sub),
                source_position  = (x_src, y_src),
                images_use_reduced_shear=True,   # imágenes con reduced shear (como tu set original)
                scale_outputs=DO_SCALE
            )

            # Almacenamiento
            dset_input[i]     = img_rgb
            dset_clean[i]     = img_clean
            dset_delta_psi[i] = delta_psi
            dset_mass[i]      = M_sub
            dset_xpos[i]      = x_sub
            dset_ypos[i]      = y_sub
            dset_srcx[i]      = x_src
            dset_srcy[i]      = y_src

        # Metadatos
        f.attrs['N_samples']       = N
        f.attrs['mass_range']      = (mass_min, mass_max)
        f.attrs['subhalo_pos_rng'] = (subhalo_pos_min, subhalo_pos_max)
        f.attrs['source_pos_rng']  = (source_pos_min, source_pos_max)
        f.attrs['pixel_scale_arcsec'] = PIXEL_SCALE
        f.attrs['stamp_size_arcsec']  = STAMP_SIZE_ARCSEC
        f.attrs['bands']           = 'g,r,i'
        f.attrs['psf']             = 'Gaussian, 10-year coadd'
        f.attrs['images_sqrt_stretch'] = bool(DO_SCALE)
        f.attrs['shear_images']    = 'SHEAR_REDUCED'
        f.attrs['shear_potential'] = 'SHEAR'
        f.attrs['z_lens_source']   = (float(z_lens), float(z_source))
        f.attrs['H0_Om0_Ob0']      = (70.0, 0.3, 0.048)
        f.attrs['description']     = ("RGB LSST-like lensing dataset: noisy-with-subhalo, clean-no-subhalo, and Δψ map. "
                                      "Labels: subhalo mass and (x,y) of subhalo and source. "
                                      "Grid for Δψ matches image grid exactly.")
