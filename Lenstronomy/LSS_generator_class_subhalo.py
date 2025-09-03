# -*- coding: utf-8 -*-
# Generador de dataset LSST con subhalos (solo subhalo, sin negativos inyectados)
# - Detectabilidad vía residuo ruidoso emparejado (espacio lineal)
# - images_clean = SIN subhalo y SIN ruido (noiseless) [guardadas con sqrt-stretch si DO_SCALE=True]
# - Se guardan además mapas de residual lineales: residual_l2_sub y residual_l2_null
# - Rango de masas: 1e4–1e9 M_sun
# - Optimizado: cache de objetos, cuantiles rápidos, chunking, paralelo opcional

import math
import numpy as np
import h5py
from joblib import Parallel, delayed

from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import FlatLambdaCDM

from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util.param_util import phi_q2_ellipticity, shear_polar2cartesian

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

# Fuente
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

# Visual
DO_SCALE = True                  # aplica sqrt-stretch a images_* guardadas (solo para visualización)
RING_HALFWIDTH = 0.30
USE_REDUCED_SHEAR = True

# -----------------------------
# Utilidades rápidas
# -----------------------------
def ring_mask(x_coords, y_coords, x0, y0, theta_E, halfwidth=0.35):
    r = np.hypot(x_coords - x0, y_coords - y0)
    return (np.abs(r - theta_E) <= halfwidth)

def sqrt_stretch_pair_fast(img_noisy, img_ref_clean):
    """Sqrt-stretch por canal usando p95 del 'clean' de referencia (rápido)."""
    v = np.quantile(img_ref_clean.reshape(-1, 3), 0.95, axis=0)
    v = np.clip(v, 1e-8, None)
    def _sqrt(x, vmax):
        x = np.clip(x, 0, vmax)
        return np.sqrt(x / vmax)
    out_noisy = np.stack([_sqrt(img_noisy[...,i], v[i]) for i in range(3)], axis=-1)
    out_clean = np.stack([_sqrt(img_ref_clean[...,i], v[i]) for i in range(3)], axis=-1)
    return out_noisy, out_clean, v

def l2_rgb(stack3):
    """||.||_2 sobre canal RGB por píxel."""
    return np.sqrt(np.sum(stack3*stack3, axis=-1))

def rms_masked(arr2, mask=None):
    """RMS de un mapa 2D; si hay máscara, aplica sobre la máscara."""
    if mask is None:
        return float(np.sqrt(np.mean(arr2*arr2)))
    sel = arr2[mask]
    if sel.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(sel*sel)))

# -----------------------------
# CACHE de render
# -----------------------------
class RenderCache:
    def __init__(self, images_use_reduced_shear=True):
        if images_use_reduced_shear:
            self.lens_list_img_sub   = ['EPL', 'SIS', 'SHEAR_REDUCED']
            self.lens_list_img_nosub = ['EPL',       'SHEAR_REDUCED']
        else:
            self.lens_list_img_sub   = ['EPL', 'SIS', 'SHEAR']
            self.lens_list_img_nosub = ['EPL',       'SHEAR']

        self.kwargs_model_sub_img = {
            'lens_model_list': self.lens_list_img_sub,
            'source_light_model_list': ['SERSIC'],
            'lens_light_model_list': [],
            'point_source_model_list': []
        }
        self.kwargs_model_nosub_img = {
            'lens_model_list': self.lens_list_img_nosub,
            'source_light_model_list': ['SERSIC'],
            'lens_light_model_list': [],
            'point_source_model_list': []
        }

        # Config banda fija
        self.band_configs = [LSST_g.kwargs_single_band(),
                             LSST_r.kwargs_single_band(),
                             LSST_i.kwargs_single_band()]
        self.sim_sub   = [SimAPI(numpix=NUMPIX, kwargs_single_band=bc, kwargs_model=self.kwargs_model_sub_img)
                          for bc in self.band_configs]
        self.sim_nosub = [SimAPI(numpix=NUMPIX, kwargs_single_band=bc, kwargs_model=self.kwargs_model_nosub_img)
                          for bc in self.band_configs]
        self.im_sub    = [sim.image_model_class(KWARGS_NUMERICS)   for sim in self.sim_sub]
        self.im_nosub  = [sim.image_model_class(KWARGS_NUMERICS)   for sim in self.sim_nosub]

        # Pixel grid fijo
        ra0 = -STAMP_SIZE_ARCSEC / 2.0
        dec0 = -STAMP_SIZE_ARCSEC / 2.0
        transform = np.eye(2) * PIXEL_SCALE
        pg = PixelGrid(nx=NUMPIX, ny=NUMPIX, ra_at_xy_0=ra0, dec_at_xy_0=dec0, transform_pix2angle=transform)
        self.x_coords, self.y_coords = pg.pixel_coordinates

        # Máscara anular
        self.ring_mask = ring_mask(self.x_coords, self.y_coords,
                                   kwargs_main['center_x'], kwargs_main['center_y'],
                                   theta_E_main, halfwidth=RING_HALFWIDTH)

        # LensModel para Δψ
        self.lm_psi_sub   = LensModel(lens_model_list=['EPL','SIS','SHEAR'])
        self.lm_psi_nosub = LensModel(lens_model_list=['EPL',      'SHEAR'])

    def render_rgb(self, use_sub, kwargs_lens, source_mags, add_noise=False, noise_seed_base=0):
        sims = self.sim_sub if use_sub else self.sim_nosub
        ims  = self.im_sub  if use_sub else self.im_nosub

        imgs = []
        for b_idx, (sim, imSim, mags) in enumerate(zip(sims, ims, source_mags)):
            _, kwargs_source, _ = sim.magnitude2amplitude([], mags, [])
            img = imSim.image(kwargs_lens, kwargs_source, None, None)
            if add_noise:
                state = np.random.get_state()
                np.random.seed(int(noise_seed_base) + b_idx)
                img = img + sim.noise_for_model(model=img)
                np.random.set_state(state)
            imgs.append(img)
        return np.stack(imgs, axis=-1)  # (H,W,3)

# -----------------------------
# Núcleo rápido
# -----------------------------
def simulate_pair_and_delta_fast(cache: RenderCache,
                                 mass_subhalo, position_subhalo, source_position,
                                 scale_outputs=DO_SCALE, snr_thresh_noisy=3.5):  # UMBRAL MÁS ALTO (por defecto 3.5)
    # SIS theta_E
    M_sub = mass_subhalo * u.M_sun
    thetaE_sub_rad = np.sqrt(4 * G * M_sub / c**2 * (D_ds / (D_d * D_s)))
    thetaE_sub = (thetaE_sub_rad * u.rad).to(u.arcsec).value

    x_sub, y_sub = position_subhalo
    kwargs_sub = {'theta_E': thetaE_sub, 'center_x': x_sub, 'center_y': y_sub}
    kwargs_lens       = [kwargs_main, kwargs_sub, kwargs_shear]
    kwargs_lens_nosub = [kwargs_main,           kwargs_shear]

    x_src, y_src = source_position
    base = {'R_sersic': R_sersic, 'n_sersic': n_sersic, 'center_x': x_src, 'center_y': y_src}
    src_g = [{**base, 'magnitude': mag_g}]
    src_r = [{**base, 'magnitude': mag_g - g_r}]
    src_i = [{**base, 'magnitude': mag_g - g_i}]
    source_mags = [src_g, src_r, src_i]

    # clean noiseless (no-sub)
    img_clean_noiseless = cache.render_rgb(False, kwargs_lens_nosub, source_mags,
                                           add_noise=False, noise_seed_base=0)

    # noisy paired A/B (lineales)
    seedA = int(rng.integers(0, 2**31-1))
    img_clean_A = cache.render_rgb(False, kwargs_lens_nosub, source_mags,
                                   add_noise=True, noise_seed_base=seedA)
    img_sub_A   = cache.render_rgb(True,  kwargs_lens,       source_mags,
                                   add_noise=True, noise_seed_base=seedA)
    img_clean_B = cache.render_rgb(False, kwargs_lens_nosub, source_mags,
                                   add_noise=True, noise_seed_base=seedA+7919)

    # Residuales (lineales)
    resid_sub  = img_sub_A   - img_clean_A
    resid_null = img_clean_B - img_clean_A
    R2_sub  = l2_rgb(resid_sub)   # (H,W)
    R2_null = l2_rgb(resid_null)  # (H,W)

    # SNR (lineal, con máscara)
    rms_sub  = rms_masked(R2_sub,  cache.ring_mask)
    rms_null = rms_masked(R2_null, cache.ring_mask)
    snr_eff  = rms_sub / (rms_null + 1e-12)
    is_det   = int(snr_eff >= snr_thresh_noisy)

    # Δψ
    x_flat, y_flat = cache.x_coords.ravel(), cache.y_coords.ravel()
    psi_sub   = cache.lm_psi_sub.potential(x_flat, y_flat, kwargs_lens).reshape(cache.x_coords.shape)
    psi_nosub = cache.lm_psi_nosub.potential(x_flat, y_flat, kwargs_lens_nosub).reshape(cache.x_coords.shape)
    delta_psi_map = psi_sub - psi_nosub
    epsilon = 1e-3
    delta_psi_log = np.log10(np.abs(delta_psi_map) + epsilon) * np.sign(delta_psi_map)

    # Visual stretch solo para guardar imágenes “bonitas”
    if scale_outputs:
        img_sub_vis,  _ , _ = sqrt_stretch_pair_fast(img_sub_A,           img_clean_A)          # noisy con sub
        _, img_clean_noiseless_vis, _ = sqrt_stretch_pair_fast(img_clean_noiseless, img_clean_noiseless)  # clean noiseless
    else:
        img_sub_vis = img_sub_A
        img_clean_noiseless_vis = img_clean_noiseless

    return (img_sub_vis.astype('f4'),
            img_clean_noiseless_vis.astype('f4'),
            delta_psi_log.astype('f4'),
            R2_sub.astype('f4'), R2_null.astype('f4'),   # <- residuales lineales para ploteo
            float(snr_eff), is_det)

# -----------------------------
# Generación del dataset (chunking + paralelo opcional)
# -----------------------------
if __name__ == "__main__":
    # Configuración dataset
    N_TOTAL = 10000
    SNR_THRESH = 10.0   # <---- UMBRAL MÁS ALTO
    out_name = 'LSST_Mmxy_onlySubhalo_1e4to1e9_noiselessClean_fast_thr3p5.h5'

    # RANGO DE MASAS
    mass_min, mass_max = 1e5, 1e9
    subhalo_pos_min, subhalo_pos_max = -1.6, 1.6
    source_pos_min,  source_pos_max  = -1.0, 1.0

    # Cache inicial
    cache = RenderCache(images_use_reduced_shear=USE_REDUCED_SHEAR)

    # Dummy para shapes
    ex = simulate_pair_and_delta_fast(cache, 1e7, (0.0,0.5), (0.0,0.0),
                                      scale_outputs=DO_SCALE, snr_thresh_noisy=SNR_THRESH)
    ex_rgb, ex_clean, ex_dpsi, ex_r2s, ex_r2n, ex_snr, ex_det = ex
    ny, nx, n_channels = ex_rgb.shape

    with h5py.File(out_name, 'w') as f:
        dset_input      = f.create_dataset('images_rgb',     (N_TOTAL, ny, nx, n_channels), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_clean      = f.create_dataset('images_clean',   (N_TOTAL, ny, nx, n_channels), dtype='f4', chunks=True, compression='gzip', compression_opts=4)  # noiseless (visual)
        dset_delta_psi  = f.create_dataset('delta_psi_maps', (N_TOTAL, ny, nx),             dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        # Residuales lineales por píxel (L2 RGB)
        dset_r2_sub     = f.create_dataset('residual_l2_sub',  (N_TOTAL, ny, nx),           dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_r2_null    = f.create_dataset('residual_l2_null', (N_TOTAL, ny, nx),           dtype='f4', chunks=True, compression='gzip', compression_opts=4)

        dset_mass       = f.create_dataset('subhalo_mass',   (N_TOTAL,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_xpos       = f.create_dataset('subhalo_x',      (N_TOTAL,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_ypos       = f.create_dataset('subhalo_y',      (N_TOTAL,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_srcx       = f.create_dataset('source_x',       (N_TOTAL,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_srcy       = f.create_dataset('source_y',       (N_TOTAL,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_snr        = f.create_dataset('snr_proxy',      (N_TOTAL,),                    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_detect     = f.create_dataset('is_detectable',  (N_TOTAL,),                    dtype='i1', chunks=True, compression='gzip', compression_opts=4)

        # Rendimiento
        CHUNK = 256
        N_CH  = math.ceil(N_TOTAL/CHUNK)
        N_JOBS = 0  # >0 para paralelo con joblib

        def _simulate_i():
            M_sub = rng.uniform(mass_min, mass_max)
            x_sub = rng.uniform(subhalo_pos_min, subhalo_pos_max)
            y_sub = rng.uniform(subhalo_pos_min, subhalo_pos_max)
            x_src = rng.uniform(source_pos_min,  source_pos_max)
            y_src = rng.uniform(source_pos_min,  source_pos_max)
            (img_rgb, img_clean_noiseless, dpsi,
             R2_sub, R2_null, snr_eff, is_det) = simulate_pair_and_delta_fast(
                 cache, M_sub, (x_sub,y_sub), (x_src,y_src),
                 scale_outputs=DO_SCALE, snr_thresh_noisy=SNR_THRESH
             )
            return img_rgb, img_clean_noiseless, dpsi, R2_sub, R2_null, snr_eff, is_det, (M_sub,x_sub,y_sub,x_src,y_src)

        for ch in range(N_CH):
            a = ch*CHUNK
            b = min(N_TOTAL, (ch+1)*CHUNK)
            n_this = b - a

            if N_JOBS and N_JOBS > 1:
                results = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(_simulate_i)() for _ in range(n_this))
            else:
                results = [_simulate_i() for _ in range(n_this)]

            # Volcar chunk
            for j, (img_rgb, img_clean_noiseless, dpsi, R2_sub, R2_null, snr_eff, is_det, meta) in enumerate(results):
                i = a + j
                M_sub, x_sub, y_sub, x_src, y_src = meta
                dset_input[i]      = img_rgb
                dset_clean[i]      = img_clean_noiseless
                dset_delta_psi[i]  = dpsi
                dset_r2_sub[i]     = R2_sub
                dset_r2_null[i]    = R2_null
                dset_mass[i]       = M_sub
                dset_xpos[i]       = x_sub
                dset_ypos[i]       = y_sub
                dset_srcx[i]       = x_src
                dset_srcy[i]       = y_src
                dset_snr[i]        = snr_eff
                dset_detect[i]     = is_det

        # Metadatos
        f.attrs['N_samples']               = N_TOTAL
        f.attrs['mass_range']              = (mass_min, mass_max)
        f.attrs['subhalo_pos_rng']         = (subhalo_pos_min, subhalo_pos_max)
        f.attrs['source_pos_rng']          = (source_pos_min, source_pos_max)
        f.attrs['pixel_scale_arcsec']      = PIXEL_SCALE
        f.attrs['stamp_size_arcsec']       = STAMP_SIZE_ARCSEC
        f.attrs['bands']                   = 'g,r,i'
        f.attrs['psf']                     = 'Gaussian, 10-year coadd'
        f.attrs['images_sqrt_stretch']     = bool(DO_SCALE)        # imágenes guardadas estiradas
        f.attrs['residual_maps_space']     = 'linear_L2_RGB'       # residual_l2_* en espacio lineal
        f.attrs['shear_images']            = 'SHEAR_REDUCED' if USE_REDUCED_SHEAR else 'SHEAR'
        f.attrs['shear_potential']         = 'SHEAR'
        f.attrs['z_lens_source']           = (float(z_lens), float(z_source))
        f.attrs['H0_Om0_Ob0']              = (70.0, 0.3, 0.048)
        f.attrs['snr_proxy_kind']          = 'paired_noisy_residual_global'
        f.attrs['snr_thresh_noisy']        = SNR_THRESH
        f.attrs['use_ring_mask']           = True
        f.attrs['ring_halfwidth']          = RING_HALFWIDTH
        f.attrs['description']             = ("Dataset LSST-like RGB (solo subhalo, FAST, thr=%.2f): "
                                              "images_rgb/clean guardadas con sqrt-stretch opcional; "
                                              "residual_l2_* en espacio lineal; detectabilidad vía residuo emparejado."
                                              % SNR_THRESH)
