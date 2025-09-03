# -*- coding: utf-8 -*-
# Generador de dataset LSST con subhalos para VAE/Clasificación/Regresión
# Versión "paired noisy residual" + NEGATIVOS sin subhalo:
#  - Pareado de ruido (misma semilla para con/sin subhalo) + nulo de referencia
#  - Métrica SNR_eff = RMS(residuo_sub) / RMS(residuo_nulo)
#  - Opción de enmascarar en un anillo alrededor de theta_E_main (para construir SNR si quieres)
#  - NUEVO: inyecta muestras sin subhalo (negativas obvias) según FRAC_NO_SUB
#  - Guarda snr_proxy, is_detectable en HDF5 (y Δψ=0 en negativos)

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
from lenstronomy.Data.pixel_grid import PixelGrid

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

# Imagen
STAMP_SIZE_ARCSEC = 6.0
PIXEL_SCALE = 0.15
NUMPIX = int(round(STAMP_SIZE_ARCSEC / PIXEL_SCALE))
KWARGS_NUMERICS = {'point_source_supersampling_factor': 1}

# Escalado visual
DO_SCALE = True

# -----------------------------
# Utilidades
# -----------------------------
def scale_pair_rgb(img_noisy, img_clean):
    """Aplica sqrt-stretch por canal usando percentil 95 de la imagen clean."""
    def _p95(image):
        flat = image.flatten()
        flat.sort()
        return flat[int(0.95 * len(flat))]
    v = np.array([_p95(img_clean[..., i]) for i in range(img_clean.shape[-1])], dtype=float)
    out_noisy = np.zeros_like(img_noisy, dtype=float)
    out_clean = np.zeros_like(img_clean, dtype=float)
    for i in range(img_clean.shape[-1]):
        out_noisy[..., i] = plot_util.sqrt(img_noisy[..., i], scale_min=0, scale_max=v[i])
        out_clean[..., i] = plot_util.sqrt(img_clean[..., i], scale_min=0, scale_max=v[i])
    return out_noisy, out_clean, v

def _image_and_noise(sim, imSim, kwargs_lens, kwargs_source, add_noise, noise_seed):
    """Devuelve imagen (H,W) y añade ruido con una semilla controlada."""
    image = imSim.image(kwargs_lens, kwargs_source, None, None)
    if add_noise:
        state = np.random.get_state()
        np.random.seed(noise_seed)
        image = image + sim.noise_for_model(model=image)
        np.random.set_state(state)
    return image

def simulate_rgb_image_seeded(band_configs, model_config, kwargs_lens, kwargs_source_mags,
                              add_noise=True, noise_seed_base=0):
    """
    Igual que simulate_rgb_image, pero controlando la semilla de ruido por banda.
    Si llamas dos veces con el MISMO noise_seed_base obtendrás el MISMO campo de ruido.
    """
    images = []
    xy_grid = None
    for b_idx, (band_config, kwargs_source_mag) in enumerate(zip(band_configs, kwargs_source_mags)):
        kw_band = band_config.kwargs_single_band()
        sim = SimAPI(numpix=NUMPIX, kwargs_single_band=kw_band, kwargs_model=model_config)
        _, kwargs_source, _ = sim.magnitude2amplitude([], kwargs_source_mag, [])
        imSim = sim.image_model_class(KWARGS_NUMERICS)

        img = _image_and_noise(sim, imSim, kwargs_lens, kwargs_source,
                               add_noise=add_noise, noise_seed=noise_seed_base + b_idx)
        images.append(img)

        if xy_grid is None:
            data_obj = getattr(imSim, 'Data', None) or getattr(imSim, 'data_class', None) or getattr(sim, 'data_class', None)
            got = False
            if data_obj is not None:
                try:
                    x_coords, y_coords = data_obj.pixel_grid.pixel_coordinates; got = True
                except Exception:
                    pass
                if (not got) and hasattr(data_obj, 'pixel_coordinates'):
                    try:
                        x_coords, y_coords = data_obj.pixel_coordinates; got = True
                    except Exception:
                        pass
                if (not got) and hasattr(data_obj, 'x_grid') and hasattr(data_obj, 'y_grid'):
                    try:
                        x_coords = np.asarray(data_obj.x_grid).reshape(NUMPIX, NUMPIX)
                        y_coords = np.asarray(data_obj.y_grid).reshape(NUMPIX, NUMPIX); got = True
                    except Exception:
                        pass
                if (not got) and hasattr(data_obj, 'grid_class') and hasattr(data_obj.grid_class, 'pixel_coordinates'):
                    try:
                        x_coords, y_coords = data_obj.grid_class.pixel_coordinates; got = True
                    except Exception:
                        pass
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

def ring_mask(x_coords, y_coords, x0, y0, theta_E, halfwidth=0.35):
    r = np.hypot(x_coords - x0, y_coords - y0)
    return (np.abs(r - theta_E) <= halfwidth)

def rms_map_l2_rgb_masked(stack3, mask=None):
    L2 = np.sqrt(np.sum(stack3**2, axis=-1))
    if mask is None or mask.sum() == 0:
        return float(np.sqrt(np.mean(L2**2)))
    L2m = L2[mask]
    return float(np.sqrt(np.mean((L2m**2))))

# -----------------------------
# Núcleos de simulación
# -----------------------------
def simulate_pair_and_delta(mass_subhalo, position_subhalo, source_position,
                            images_use_reduced_shear=True, scale_outputs=DO_SCALE,
                            snr_thresh_noisy=1.6, use_ring_mask=True, ring_halfwidth=0.30):
    """
    Caso CON subhalo. Devuelve:
      img_sub (noisy A), img_clean (noisy A), delta_psi_log, snr_eff, is_detectable
    """
    # --- Subhalo SIS ---
    M_sub = mass_subhalo * u.M_sun
    thetaE_sub_rad = np.sqrt(4 * G * M_sub / c**2 * (D_ds / (D_d * D_s)))
    thetaE_sub = (thetaE_sub_rad * u.rad).to(u.arcsec).value

    x_sub, y_sub = position_subhalo
    kwargs_sub = {'theta_E': thetaE_sub, 'center_x': x_sub, 'center_y': y_sub}
    kwargs_lens       = [kwargs_main, kwargs_sub, kwargs_shear]
    kwargs_lens_nosub = [kwargs_main,           kwargs_shear]

    # --- Modelos ---
    if images_use_reduced_shear:
        lens_list_img_sub   = ['EPL', 'SIS', 'SHEAR_REDUCED']
        lens_list_img_nosub = ['EPL',       'SHEAR_REDUCED']
    else:
        lens_list_img_sub   = ['EPL', 'SIS', 'SHEAR']
        lens_list_img_nosub = ['EPL',       'SHEAR']

    kwargs_model_sub_img = {'lens_model_list': lens_list_img_sub,
                            'source_light_model_list': ['SERSIC'],
                            'lens_light_model_list': [],
                            'point_source_model_list': []}
    kwargs_model_nosub_img = {'lens_model_list': lens_list_img_nosub,
                              'source_light_model_list': ['SERSIC'],
                              'lens_light_model_list': [],
                              'point_source_model_list': []}

    # --- Fuente ---
    x_src, y_src = source_position
    base = {'R_sersic': R_sersic, 'n_sersic': n_sersic, 'center_x': x_src, 'center_y': y_src}
    src_g = [{**base, 'magnitude': mag_g}]
    src_r = [{**base, 'magnitude': mag_g - g_r}]
    src_i = [{**base, 'magnitude': mag_g - g_i}]
    source_mags = [src_g, src_r, src_i]

    # --- Noiseless para Δψ y máscara ---
    img_clean_noiseless, (x_coords, y_coords) = simulate_rgb_image_seeded(
        lsst_bands, kwargs_model_nosub_img, kwargs_lens_nosub, source_mags, add_noise=False
    )

    # --- Imágenes NOISY emparejadas ---
    noise_seed_base = rng.integers(0, 2**31 - 1, dtype=np.int64)
    img_clean_A, _ = simulate_rgb_image_seeded(
        lsst_bands, kwargs_model_nosub_img, kwargs_lens_nosub, source_mags,
        add_noise=True, noise_seed_base=int(noise_seed_base)
    )
    img_sub_A, _ = simulate_rgb_image_seeded(
        lsst_bands, kwargs_model_sub_img, kwargs_lens, source_mags,
        add_noise=True, noise_seed_base=int(noise_seed_base)
    )
    img_clean_B, _ = simulate_rgb_image_seeded(
        lsst_bands, kwargs_model_nosub_img, kwargs_lens_nosub, source_mags,
        add_noise=True, noise_seed_base=int(noise_seed_base + 7919)
    )

    # --- Residuales / SNR ---
    resid_sub  = img_sub_A   - img_clean_A
    resid_null = img_clean_B - img_clean_A
    mask = ring_mask(x_coords, y_coords, kwargs_main['center_x'], kwargs_main['center_y'],
                     theta_E_main, halfwidth=ring_halfwidth) if use_ring_mask else None

    rms_sub  = rms_map_l2_rgb_masked(resid_sub,  mask)
    rms_null = rms_map_l2_rgb_masked(resid_null, mask)
    snr_eff  = rms_sub / (rms_null + 1e-12)
    is_det   = bool(snr_eff >= snr_thresh_noisy)

    # --- Potencial Δψ ---
    x_flat, y_flat = x_coords.ravel(), y_coords.ravel()
    lensModel_psi       = LensModel(lens_model_list=['EPL','SIS','SHEAR'])
    lensModel_nosub_psi = LensModel(lens_model_list=['EPL',      'SHEAR'])
    psi_sub   = lensModel_psi.potential(x_flat, y_flat, kwargs_lens).reshape(x_coords.shape)
    psi_nosub = lensModel_nosub_psi.potential(x_flat, y_flat, kwargs_lens_nosub).reshape(x_coords.shape)
    delta_psi_map = psi_sub - psi_nosub
    epsilon = 1e-3
    delta_psi_log = np.log10(np.abs(delta_psi_map) + epsilon) * np.sign(delta_psi_map)

    # --- Salidas visuales (lo que guardamos) ---
    if scale_outputs:
        img_sub_vis, img_clean_vis, _ = scale_pair_rgb(img_sub_A, img_clean_A)
    else:
        img_sub_vis, img_clean_vis = img_sub_A, img_clean_A

    return img_sub_vis, img_clean_vis, delta_psi_log, float(snr_eff), int(is_det)

def simulate_negative_no_subhalo(source_position,
                                 images_use_reduced_shear=True,
                                 scale_outputs=DO_SCALE):
    """
    Caso SIN subhalo (negativo 'obvio'):
      images_rgb = noisy-no-subhalo (ruido A)
      images_clean = noisy-no-subhalo (mismo A, para consistencia)
      Δψ = 0, snr_proxy = 0, is_detectable = 0, mass=0, x/y = NaN
    """
    # Modelos sin subhalo
    lens_list_img = ['EPL','SHEAR_REDUCED'] if images_use_reduced_shear else ['EPL','SHEAR']
    kwargs_model_img = {'lens_model_list': lens_list_img,
                        'source_light_model_list': ['SERSIC'],
                        'lens_light_model_list': [],
                        'point_source_model_list': []}

    # Fuente
    x_src, y_src = source_position
    base = {'R_sersic': R_sersic, 'n_sersic': n_sersic, 'center_x': x_src, 'center_y': y_src}
    src_g = [{**base, 'magnitude': mag_g}]
    src_r = [{**base, 'magnitude': mag_g - g_r}]
    src_i = [{**base, 'magnitude': mag_g - g_i}]
    source_mags = [src_g, src_r, src_i]

    # Imágenes ruidosas (A) sin subhalo
    kwargs_lens_nosub = [kwargs_main, kwargs_shear]
    noise_seed_base = rng.integers(0, 2**31 - 1, dtype=np.int64)
    img_A, (x_coords, y_coords) = simulate_rgb_image_seeded(
        lsst_bands, kwargs_model_img, kwargs_lens_nosub, source_mags,
        add_noise=True, noise_seed_base=int(noise_seed_base)
    )

    # Salidas
    delta_psi_log = np.zeros_like(x_coords, dtype='float32')  # (H,W)
    if scale_outputs:
        img_vis, _clean_vis, _ = scale_pair_rgb(img_A, img_A)
        img_clean_vis = img_vis
    else:
        img_vis = img_A
        img_clean_vis = img_A

    snr_eff = 0.0
    is_det  = 0
    return img_vis, img_clean_vis, delta_psi_log, snr_eff, is_det
# -----------------------------
# Generación del dataset
# -----------------------------
if __name__ == "__main__":
    # Configuración dataset
    N_TOTAL = 5000                 # tamaño total del dataset
    FRAC_NO_SUB = 0.25            # fracción de negativos "sin subhalo"
    N_NEG = int(round(FRAC_NO_SUB * N_TOTAL))
    N_POS = N_TOTAL - N_NEG

    mass_min, mass_max = 1e7, 1e9
    subhalo_pos_min, subhalo_pos_max = -1.6, 1.6
    source_pos_min,  source_pos_max  = -1.0, 1.0

    # Muestra dummy para shapes
    ex_rgb, ex_clean, ex_dpsi, ex_snr, ex_det = simulate_pair_and_delta(
        mass_subhalo=1e8,
        position_subhalo=(0.0, 0.5),
        source_position=(0.0, 0.0),
        images_use_reduced_shear=True,
        scale_outputs=DO_SCALE,
        snr_thresh_noisy=1.6,
        use_ring_mask=True,
        ring_halfwidth=0.30
    )
    ny, nx, n_channels = ex_rgb.shape

    with h5py.File('LSST_Mmxy_noisyResidual_paired_withNoSub.h5', 'w') as f:
        dset_input     = f.create_dataset('images_rgb',     (N_TOTAL, ny, nx, n_channels), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_clean     = f.create_dataset('images_clean',   (N_TOTAL, ny, nx, n_channels), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_delta_psi = f.create_dataset('delta_psi_maps', (N_TOTAL, ny, nx),             dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_mass      = f.create_dataset('subhalo_mass',   (N_TOTAL,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_xpos      = f.create_dataset('subhalo_x',      (N_TOTAL,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_ypos      = f.create_dataset('subhalo_y',      (N_TOTAL,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_srcx      = f.create_dataset('source_x',       (N_TOTAL,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_srcy      = f.create_dataset('source_y',       (N_TOTAL,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_snr       = f.create_dataset('snr_proxy',      (N_TOTAL,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_detect    = f.create_dataset('is_detectable',  (N_TOTAL,),                    dtype='i1', chunks=True, compression='gzip', compression_opts=4)

        # --- NEGATIVOS sin subhalo ---
        for i in range(N_NEG):
            x_src = rng.uniform(source_pos_min, source_pos_max)
            y_src = rng.uniform(source_pos_min, source_pos_max)

            img_rgb, img_clean, delta_psi, snr_proxy, is_det = simulate_negative_no_subhalo(
                source_position=(x_src, y_src),
                images_use_reduced_shear=True,
                scale_outputs=DO_SCALE
            )

            dset_input[i]     = img_rgb
            dset_clean[i]     = img_clean
            dset_delta_psi[i] = delta_psi
            dset_mass[i]      = 0.0
            dset_xpos[i]      = np.nan
            dset_ypos[i]      = np.nan
            dset_srcx[i]      = x_src
            dset_srcy[i]      = y_src
            dset_snr[i]       = snr_proxy
            dset_detect[i]    = is_det

        # --- POSITIVOS/NEGATIVOS con subhalo (etiqueta vía SNR_eff) ---
        for i in range(N_NEG, N_TOTAL):
            M_sub = rng.uniform(mass_min, mass_max)
            x_sub = rng.uniform(subhalo_pos_min, subhalo_pos_max)
            y_sub = rng.uniform(subhalo_pos_min, subhalo_pos_max)
            x_src = rng.uniform(source_pos_min, source_pos_max)
            y_src = rng.uniform(source_pos_min, source_pos_max)

            img_rgb, img_clean, delta_psi, snr_proxy, is_det = simulate_pair_and_delta(
                mass_subhalo     = M_sub,
                position_subhalo = (x_sub, y_sub),
                source_position  = (x_src, y_src),
                images_use_reduced_shear=True,
                scale_outputs=DO_SCALE,
                snr_thresh_noisy=1.6,
                use_ring_mask=True,
                ring_halfwidth=0.30
            )

            dset_input[i]     = img_rgb
            dset_clean[i]     = img_clean
            dset_delta_psi[i] = delta_psi
            dset_mass[i]      = M_sub
            dset_xpos[i]      = x_sub
            dset_ypos[i]      = y_sub
            dset_srcx[i]      = x_src
            dset_srcy[i]      = y_src
            dset_snr[i]       = snr_proxy
            dset_detect[i]    = is_det

        # Metadatos
        f.attrs['N_samples']             = N_TOTAL
        f.attrs['frac_no_sub']           = float(FRAC_NO_SUB)
        f.attrs['mass_range']            = (mass_min, mass_max)
        f.attrs['subhalo_pos_rng']       = (subhalo_pos_min, subhalo_pos_max)
        f.attrs['source_pos_rng']        = (source_pos_min, source_pos_max)
        f.attrs['pixel_scale_arcsec']    = PIXEL_SCALE
        f.attrs['stamp_size_arcsec']     = STAMP_SIZE_ARCSEC
        f.attrs['bands']                 = 'g,r,i'
        f.attrs['psf']                   = 'Gaussian, 10-year coadd'
        f.attrs['images_sqrt_stretch']   = bool(DO_SCALE)
        f.attrs['shear_images']          = 'SHEAR_REDUCED'
        f.attrs['shear_potential']       = 'SHEAR'
        f.attrs['z_lens_source']         = (float(z_lens), float(z_source))
        f.attrs['H0_Om0_Ob0']            = (70.0, 0.3, 0.048)
        f.attrs['snr_proxy_kind']        = 'paired_noisy_residual_global'
        f.attrs['snr_thresh_noisy']      = 1.6
        f.attrs['use_ring_mask']         = True
        f.attrs['ring_halfwidth']        = 0.30
        f.attrs['description']           = ("Dataset LSST-like RGB: "
                                            "images_rgb = con subhalo (ruido A) o sin subhalo (negativos inyectados), "
                                            "images_clean = no-subhalo (ruido A), Δψ; "
                                            "detectabilidad via residuo ruidoso emparejado y nulo de referencia.")
