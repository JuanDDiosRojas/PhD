# -*- coding: utf-8 -*-
# Generador de dataset LSST con subhalos para VAE/Regresión
# Versión "noisy residual" robusta (emparejada de verdad):
#  - Ruido de fondo gaussiano controlado por semilla (sin Poisson dependiente del flujo)
#  - Pareado exacto (misma semilla para subhalo y clean A)
#  - Nulo de referencia (clean B con semilla distinta)
#  - Métrica SNR_eff = RMS_L2(resid_sub) / RMS_L2(resid_null)
#  - Opción de anillo alrededor de theta_E_main
#  - Guarda snr_proxy, is_detectable, Δψ en HDF5

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

# Parámetros de imagen
STAMP_SIZE_ARCSEC = 6.0     # tamaño del recorte (lado), en arcsec
PIXEL_SCALE = 0.15          # arcsec/pixel (-> 40x40)
NUMPIX = int(round(STAMP_SIZE_ARCSEC / PIXEL_SCALE))
KWARGS_NUMERICS = {'point_source_supersampling_factor': 1}

# ¿Guardar imágenes con sqrt-stretch (true) o lineales (false)?
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


def _get_xy_from_imSim_or_fallback(imSim, sim):
    """Recupera (x,y) del grid aunque cambie la versión de lenstronomy."""
    data_obj = getattr(imSim, 'Data', None) or getattr(imSim, 'data_class', None) or getattr(sim, 'data_class', None)
    if data_obj is not None:
        try:
            return data_obj.pixel_grid.pixel_coordinates
        except Exception:
            pass
        if hasattr(data_obj, 'pixel_coordinates'):
            try:
                return data_obj.pixel_coordinates
            except Exception:
                pass
        if hasattr(data_obj, 'x_grid') and hasattr(data_obj, 'y_grid'):
            try:
                x = np.asarray(data_obj.x_grid).reshape(NUMPIX, NUMPIX)
                y = np.asarray(data_obj.y_grid).reshape(NUMPIX, NUMPIX)
                return x, y
            except Exception:
                pass
        if hasattr(data_obj, 'grid_class') and hasattr(data_obj.grid_class, 'pixel_coordinates'):
            try:
                return data_obj.grid_class.pixel_coordinates
            except Exception:
                pass
    # Fallback
    ra0 = -STAMP_SIZE_ARCSEC / 2.0
    dec0 = -STAMP_SIZE_ARCSEC / 2.0
    transform = np.eye(2) * PIXEL_SCALE
    pg = PixelGrid(nx=NUMPIX, ny=NUMPIX,
                   ra_at_xy_0=ra0, dec_at_xy_0=dec0,
                   transform_pix2angle=transform)
    return pg.pixel_coordinates


def estimate_background_sigma_per_band(lsst_bands):
    """Sigma (std) del ruido de fondo por banda (sin Poisson de fuente)."""
    sigmas = []
    zeros = np.zeros((NUMPIX, NUMPIX), dtype=float)
    for bcfg in lsst_bands:
        kw_band = bcfg.kwargs_single_band()
        sim = SimAPI(numpix=NUMPIX, kwargs_single_band=kw_band,
                     kwargs_model={'lens_model_list': [],
                                   'source_light_model_list': [],
                                   'lens_light_model_list': [],
                                   'point_source_model_list': []})
        n = sim.noise_for_model(model=zeros)   # sólo fondo instrumental/sky
        sigmas.append(float(np.std(n)))
    return np.array(sigmas, dtype=float)  # (3,)


def simulate_rgb_image_noiseless(band_configs, model_config, kwargs_lens, kwargs_source_mags):
    """Imágenes noiseless (g,r,i) + grid (x,y)."""
    images = []
    xy_grid = None
    for band_config, kwargs_source_mag in zip(band_configs, kwargs_source_mags):
        kw_band = band_config.kwargs_single_band()
        sim = SimAPI(numpix=NUMPIX, kwargs_single_band=kw_band, kwargs_model=model_config)
        _, kwargs_source, _ = sim.magnitude2amplitude([], kwargs_source_mag, [])
        imSim = sim.image_model_class(KWARGS_NUMERICS)
        image = imSim.image(kwargs_lens, kwargs_source, None, None)
        images.append(image)
        if xy_grid is None:
            xy_grid = _get_xy_from_imSim_or_fallback(imSim, sim)
    return np.stack(images, axis=-1), xy_grid


def add_seeded_background_noise(stack3, sigma_per_band, seed_base):
    """
    Añade ruido gaussiano de fondo por banda con semilla controlada.
    Esto hace que subhalo y clean_A compartan EXACTAMENTE el mismo ruido.
    """
    noisy = np.empty_like(stack3, dtype=float)
    H, W, B = stack3.shape
    for b in range(B):
        rng_b = np.random.default_rng(int(seed_base) + b)
        noise = rng_b.normal(loc=0.0, scale=float(sigma_per_band[b]), size=(H, W))
        noisy[..., b] = stack3[..., b] + noise
    return noisy


def ring_mask(x_coords, y_coords, x0, y0, theta_E, halfwidth=0.35):
    r = np.hypot(x_coords - x0, y_coords - y0)
    return (np.abs(r - theta_E) <= halfwidth)


def rms_map_l2_rgb_masked(stack3, mask):
    L2 = np.sqrt(np.sum(stack3**2, axis=-1))  # norma L2 por píxel en RGB
    if mask is None or mask.sum() == 0:
        return float(np.sqrt(np.mean(L2**2)))
    return float(np.sqrt(np.mean((L2[mask]**2))))


# -----------------------------
# Núcleo: una muestra (imágenes + Δψ + detectabilidad-noisy emparejada)
# -----------------------------
def simulate_pair_and_delta(mass_subhalo, position_subhalo, source_position,
                            images_use_reduced_shear=True, scale_outputs=DO_SCALE,
                            snr_thresh_noisy=1.6, use_ring_mask=True, ring_halfwidth=0.35):
    """
    Devuelve:
      img_sub (noisy emparejada), img_clean (noisy emparejada),
      delta_psi_log, snr_eff, is_detectable
    """
    # --- Subhalo SIS ---
    M_sub = mass_subhalo * u.M_sun
    thetaE_sub_rad = np.sqrt(4 * G * M_sub / c**2 * (D_ds / (D_d * D_s)))
    thetaE_sub = (thetaE_sub_rad * u.rad).to(u.arcsec).value

    x_sub, y_sub = position_subhalo
    kwargs_sub = {'theta_E': thetaE_sub, 'center_x': x_sub, 'center_y': y_sub}
    kwargs_lens       = [kwargs_main, kwargs_sub, kwargs_shear]
    kwargs_lens_nosub = [kwargs_main,           kwargs_shear]

    # --- Modelos de imagen ---
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
    kwargs_source_mag_g = [{**base, 'magnitude': mag_g}]
    kwargs_source_mag_r = [{**base, 'magnitude': mag_g - g_r}]
    kwargs_source_mag_i = [{**base, 'magnitude': mag_g - g_i}]
    source_mags = [kwargs_source_mag_g, kwargs_source_mag_r, kwargs_source_mag_i]

    # --- Imágenes noiseless (para Δψ y máscara) ---
    img_clean_0, (x_coords, y_coords) = simulate_rgb_image_noiseless(
        lsst_bands, kwargs_model_nosub_img, kwargs_lens_nosub, source_mags
    )
    img_sub_0, _ = simulate_rgb_image_noiseless(
        lsst_bands, kwargs_model_sub_img, kwargs_lens, source_mags
    )

    # --- Sigma de fondo por banda (constante por banda) ---
    sigma_bkg_per_band = estimate_background_sigma_per_band(lsst_bands)

    # --- Ruido emparejado (A) y nulo de referencia (B) ---
    seed_base = rng.integers(0, 2**31 - 1, dtype=np.int64)
    img_clean_A = add_seeded_background_noise(img_clean_0, sigma_bkg_per_band, seed_base)
    img_sub_A   = add_seeded_background_noise(img_sub_0,   sigma_bkg_per_band, seed_base)         # <- mismo ruido
    img_clean_B = add_seeded_background_noise(img_clean_0, sigma_bkg_per_band, int(seed_base)+7919) # <- ruido distinto

    # --- Residuales y SNR efectivo ---
    resid_sub  = img_sub_A   - img_clean_A   # contiene casi sólo señal del subhalo
    resid_null = img_clean_B - img_clean_A   # baseline de "sólo ruido"

    mask = ring_mask(x_coords, y_coords, kwargs_main['center_x'], kwargs_main['center_y'],
                     theta_E_main, halfwidth=ring_halfwidth) if use_ring_mask else None

    rms_sub  = rms_map_l2_rgb_masked(resid_sub,  mask)
    rms_null = rms_map_l2_rgb_masked(resid_null, mask)
    snr_eff  = rms_sub / (rms_null + 1e-12)
    is_det   = bool(snr_eff >= snr_thresh_noisy)

    # --- Potencial en el MISMO grid ---
    x_flat, y_flat = x_coords.ravel(), y_coords.ravel()
    lensModel_psi       = LensModel(lens_model_list=['EPL','SIS','SHEAR'])
    lensModel_nosub_psi = LensModel(lens_model_list=['EPL','SHEAR'])
    psi_sub   = lensModel_psi.potential(x_flat, y_flat, kwargs_lens).reshape(x_coords.shape)
    psi_nosub = lensModel_nosub_psi.potential(x_flat, y_flat, kwargs_lens_nosub).reshape(x_coords.shape)
    delta_psi_map = psi_sub - psi_nosub
    epsilon = 1e-3
    delta_psi_log = np.log10(np.abs(delta_psi_map) + epsilon) * np.sign(delta_psi_map)

    # --- Salidas visuales (guardamos las NOISY emparejadas) ---
    if scale_outputs:
        img_sub_vis, img_clean_vis, _ = scale_pair_rgb(img_sub_A, img_clean_A)
    else:
        img_sub_vis, img_clean_vis = img_sub_A, img_clean_A

    return img_sub_vis, img_clean_vis, delta_psi_log, float(snr_eff), int(is_det)


# -----------------------------
# Generación del dataset
# -----------------------------
if __name__ == "__main__":
    # Configuración dataset
    N = 500
    mass_min, mass_max = 1e7, 1e9
    subhalo_pos_min, subhalo_pos_max = -1.6, 1.6
    source_pos_min,  source_pos_max  = -1.1, 1.1

    # Ejemplo para dimensiones
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

    # Archivo HDF5
    with h5py.File('LSST_Mmxy_noisyResidual_paired2.h5', 'w') as f:
        # Datasets (chunked + compresión)
        dset_input     = f.create_dataset('images_rgb',     (N, ny, nx, n_channels), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_clean     = f.create_dataset('images_clean',   (N, ny, nx, n_channels), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_delta_psi = f.create_dataset('delta_psi_maps', (N, ny, nx),             dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_mass      = f.create_dataset('subhalo_mass',   (N,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_xpos      = f.create_dataset('subhalo_x',      (N,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_ypos      = f.create_dataset('subhalo_y',      (N,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_srcx      = f.create_dataset('source_x',       (N,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_srcy      = f.create_dataset('source_y',       (N,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)

        # Detectabilidad (residual ruidoso emparejado)
        dset_snr       = f.create_dataset('snr_proxy',      (N,), dtype='f4',  chunks=True, compression='gzip', compression_opts=4)
        dset_detect    = f.create_dataset('is_detectable',  (N,), dtype='i1',  chunks=True, compression='gzip', compression_opts=4)

        for i in range(N):
            # Masa (puedes cambiar a log-uniforme si prefieres)
            M_sub = rng.uniform(mass_min, mass_max)

            # Posición del subhalo
            x_sub = rng.uniform(subhalo_pos_min, subhalo_pos_max)
            y_sub = rng.uniform(subhalo_pos_min, subhalo_pos_max)

            # Posición de la fuente
            x_src = rng.uniform(source_pos_min, source_pos_max)
            y_src = rng.uniform(source_pos_min, source_pos_max)

            # Simulación
            img_rgb, img_clean, delta_psi, snr_proxy, is_detectable = simulate_pair_and_delta(
                mass_subhalo     = M_sub,
                position_subhalo = (x_sub, y_sub),
                source_position  = (x_src, y_src),
                images_use_reduced_shear=True,
                scale_outputs=DO_SCALE,
                snr_thresh_noisy=1.6,
                use_ring_mask=True,
                ring_halfwidth=0.30
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
            dset_snr[i]       = snr_proxy
            dset_detect[i]    = is_detectable

        # Metadatos
        f.attrs['N_samples']          = N
        f.attrs['mass_range']         = (mass_min, mass_max)
        f.attrs['subhalo_pos_rng']    = (subhalo_pos_min, subhalo_pos_max)
        f.attrs['source_pos_rng']     = (source_pos_min, source_pos_max)
        f.attrs['pixel_scale_arcsec'] = PIXEL_SCALE
        f.attrs['stamp_size_arcsec']  = STAMP_SIZE_ARCSEC
        f.attrs['bands']              = 'g,r,i'
        f.attrs['psf']                = 'Gaussian, 10-year coadd'
        f.attrs['images_sqrt_stretch']= bool(DO_SCALE)
        f.attrs['shear_images']       = 'SHEAR_REDUCED'
        f.attrs['shear_potential']    = 'SHEAR'
        f.attrs['z_lens_source']      = (float(z_lens), float(z_source))
        f.attrs['H0_Om0_Ob0']         = (70.0, 0.3, 0.048)

        # Detectabilidad
        f.attrs['snr_proxy_kind']     = 'paired_noisy_residual_global (background-only noise)'
        f.attrs['snr_thresh_noisy']   = 1.6
        f.attrs['use_ring_mask']      = True
        f.attrs['ring_halfwidth']     = 0.30
        f.attrs['description']        = ("Dataset LSST-like RGB: images_rgb = noisy-with-subhalo (ruido emparejado), "
                                         "images_clean = noisy-no-subhalo (misma semilla), Δψ map; snr_proxy basado en "
                                         "residuo emparejado y nulo de referencia (clean B con semilla distinta).")
