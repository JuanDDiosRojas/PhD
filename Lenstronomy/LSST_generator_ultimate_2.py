# # LSST_generator_lenstronomy_sigma_v2.py
# # Genera dataset LSST-like con σ de lenstronomy, SNR y χ² coherentes (sin máscara) y guarda source_rgb.
# import math, numpy as np, h5py
# from joblib import Parallel, delayed
# from astropy import units as u
# from astropy.constants import G, c
# from astropy.cosmology import FlatLambdaCDM
# from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
# from lenstronomy.SimulationAPI.sim_api import SimAPI
# from lenstronomy.Data.pixel_grid import PixelGrid
# from lenstronomy.LensModel.lens_model import LensModel
# from lenstronomy.Util.param_util import phi_q2_ellipticity, shear_polar2cartesian

# # -----------------------------
# # Configuración global
# # -----------------------------
# SEED = 12345
# rng = np.random.default_rng(SEED)

# # Cosmología
# z_lens, z_source = 0.881, 2.059
# cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048)
# D_d  = cosmo.angular_diameter_distance(z_lens)
# D_s  = cosmo.angular_diameter_distance(z_source)
# D_ds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)

# # Bandas LSST (coadd 10 años, PSF gaussiana)
# LSST_g = LSST(band='g', psf_type='GAUSSIAN', coadd_years=10)
# LSST_r = LSST(band='r', psf_type='GAUSSIAN', coadd_years=10)
# LSST_i = LSST(band='i', psf_type='GAUSSIAN', coadd_years=10)

# # Fuente (Sérsic)
# mag_g = 22.0
# g_r, g_i = 1.0, 2.0       # g-r, g-i
# R_sersic, n_sersic = 0.15, 1.0

# # Lente principal (EPL) y cizalla
# theta_E_main, gamma_main = 1.452, 1.9
# e1_main, e2_main = phi_q2_ellipticity(phi=np.deg2rad(-22.29), q=0.866)
# kwargs_main = dict(theta_E=theta_E_main, gamma=gamma_main,
#                    e1=e1_main, e2=e2_main, center_x=0.0, center_y=-0.1)
# g1, g2 = shear_polar2cartesian(phi=np.deg2rad(107.9), gamma=0.015)
# kwargs_shear = dict(gamma1=g1, gamma2=g2)

# # Imagen
# STAMP_SIZE_ARCSEC = 6.0
# PIXEL_SCALE = 0.15
# NUMPIX = int(round(STAMP_SIZE_ARCSEC / PIXEL_SCALE))
# KWARGS_NUMERICS = {'point_source_supersampling_factor': 1}
# USE_REDUCED_SHEAR = True

# # Score (opcional) desde log10(chi2_red)
# SCORE_A, SCORE_B = 2.0, 1.0
# def chi2_to_score_sigmoid(chi2_red, a=SCORE_A, b=SCORE_B):
#     x = np.log10(np.clip(chi2_red, 1e-12, 1e20))
#     return 1.0 / (1.0 + np.exp(-a * (x - b)))

# # -----------------------------
# # Helpers
# # -----------------------------
# def sqrt_stretch_rgb(img_gri, p=95):
#     """Devuelve composite RGB (R<-i, G<-r, B<-g) con sqrt-stretch por percentil p (visual)."""
#     g, r, i = img_gri[...,0], img_gri[...,1], img_gri[...,2]
#     def pclip(a, p): 
#         flat=a.reshape(-1); k=int(p/100*(flat.size-1)); 
#         return float(np.partition(flat, max(0,min(k,flat.size-1)))[k]) if flat.size else 1.0
#     gmax, rmax, imax = max(pclip(g,p),1e-9), max(pclip(r,p),1e-9), max(pclip(i,p),1e-9)
#     R = np.sqrt(np.clip(i/imax, 0, 1))
#     G = np.sqrt(np.clip(r/rmax, 0, 1))
#     B = np.sqrt(np.clip(g/gmax, 0, 1))
#     return np.stack([R,G,B], axis=-1).astype('f4')

# def l2_rgb(stack3):
#     return np.sqrt(np.sum(stack3*stack3, axis=-1))  # (H,W)

# def rms_global(arr2):
#     return float(np.sqrt(np.mean(arr2*arr2)))

# def SIS_thetaE_arcsec(M_sub_Msun):
#     M = M_sub_Msun * u.M_sun
#     thetaE_rad = np.sqrt(4 * G * M / c**2 * (D_ds / (D_d * D_s)))
#     return (thetaE_rad * u.rad).to(u.arcsec).value

# def safe_log10(x): 
#     return np.log10(np.clip(np.asarray(x,float), 1e-12, 1e20))

# # -----------------------------
# # CACHE de render
# # -----------------------------
# class RenderCache:
#     def __init__(self, use_reduced_shear=True):
#         if use_reduced_shear:
#             lens_list_sub   = ['EPL', 'SIS', 'SHEAR_REDUCED']
#             lens_list_clean = ['EPL',       'SHEAR_REDUCED']
#         else:
#             lens_list_sub   = ['EPL', 'SIS', 'SHEAR']
#             lens_list_clean = ['EPL',       'SHEAR']

#         self.kwargs_model_sub = {
#             'lens_model_list': lens_list_sub,
#             'source_light_model_list': ['SERSIC'],
#             'lens_light_model_list': [],
#             'point_source_model_list': []
#         }
#         self.kwargs_model_clean = {
#             'lens_model_list': lens_list_clean,
#             'source_light_model_list': ['SERSIC'],
#             'lens_light_model_list': [],
#             'point_source_model_list': []
#         }
#         self.kwargs_model_source = {
#             'lens_model_list': [],
#             'source_light_model_list': ['SERSIC'],
#             'lens_light_model_list': [],
#             'point_source_model_list': []
#         }

#         # Por banda
#         self.band_cfgs = [LSST_g.kwargs_single_band(),
#                           LSST_r.kwargs_single_band(),
#                           LSST_i.kwargs_single_band()]
#         self.sim_sub    = [SimAPI(numpix=NUMPIX, kwargs_single_band=bc, kwargs_model=self.kwargs_model_sub)   for bc in self.band_cfgs]
#         self.sim_clean  = [SimAPI(numpix=NUMPIX, kwargs_single_band=bc, kwargs_model=self.kwargs_model_clean) for bc in self.band_cfgs]
#         self.sim_source = [SimAPI(numpix=NUMPIX, kwargs_single_band=bc, kwargs_model=self.kwargs_model_source)for bc in self.band_cfgs]

#         self.im_sub    = [sim.image_model_class(KWARGS_NUMERICS)   for sim in self.sim_sub]
#         self.im_clean  = [sim.image_model_class(KWARGS_NUMERICS)   for sim in self.sim_clean]
#         self.im_source = [sim.image_model_class(KWARGS_NUMERICS)   for sim in self.sim_source]

#         # Grid común
#         ra0 = -STAMP_SIZE_ARCSEC/2.0; dec0 = -STAMP_SIZE_ARCSEC/2.0
#         transform = np.eye(2)*PIXEL_SCALE
#         pg = PixelGrid(nx=NUMPIX, ny=NUMPIX, ra_at_xy_0=ra0, dec_at_xy_0=dec0, transform_pix2angle=transform)
#         self.x_coords, self.y_coords = pg.pixel_coordinates

#         # Para Δψ
#         self.lm_psi_sub   = LensModel(lens_model_list=['EPL','SIS','SHEAR'])
#         self.lm_psi_clean = LensModel(lens_model_list=['EPL',      'SHEAR'])

#     def render_one_band(self, b, kwargs_lens, kwargs_lens_nosub, source_mags, seedA, seedB):
#         sim_c, im_c = self.sim_clean[b], self.im_clean[b]
#         sim_s, im_s = self.sim_sub[b],   self.im_sub[b]
#         sim_src, im_src = self.sim_source[b], self.im_source[b]

#         # Modelo (sin ruido) clean/sub/source
#         _, kw_src, _ = sim_c.magnitude2amplitude([], source_mags[b], [])
#         img_clean_model = im_c.image(kwargs_lens_nosub, kw_src, None, None)
#         img_sub_model   = im_s.image(kwargs_lens,       kw_src, None, None)
#         img_src_model   = im_src.image([],              kw_src, None, None)  # solo fuente

#         # σ por banda (estimado con 2 draws de lenstronomy): Var ≈ 0.5 * (n1-n2)^2
#         #   -> sigma = |n1 - n2| / sqrt(2)
#         def sigma_from_two_draws(sim, model_img, seed0):
#             state = np.random.get_state()
#             np.random.seed(int(seed0))
#             n1 = sim.noise_for_model(model=model_img)
#             np.random.seed(int(seed0)+7919)
#             n2 = sim.noise_for_model(model=model_img)
#             np.random.set_state(state)
#             return np.abs(n1 - n2)/np.sqrt(2.0)

#         sigma_clean = sigma_from_two_draws(sim_c, img_clean_model, seedA)
#         sigma_sub   = sigma_from_two_draws(sim_s, img_sub_model,   seedA)

#         # Noisy A (pareado) para SNR
#         state = np.random.get_state()
#         np.random.seed(int(seedA))
#         img_clean_A = img_clean_model + sim_c.noise_for_model(model=img_clean_model)
#         np.random.seed(int(seedA))
#         img_sub_A   = img_sub_model   + sim_s.noise_for_model(model=img_sub_model)
#         np.random.set_state(state)

#         # Noisy B (independiente) para nulo y χ²
#         state = np.random.get_state()
#         np.random.seed(int(seedB))
#         img_clean_B = img_clean_model + sim_c.noise_for_model(model=img_clean_model)
#         np.random.set_state(state)

#         return (img_clean_model.astype('f4'), img_sub_model.astype('f4'), img_src_model.astype('f4'),
#                 img_clean_A.astype('f4'), img_sub_A.astype('f4'), img_clean_B.astype('f4'),
#                 sigma_clean.astype('f4'), sigma_sub.astype('f4'))

# # -----------------------------
# # Núcleo: una simulación
# # -----------------------------
# def simulate_pair_fast(cache: RenderCache, M_sub, pos_sub, pos_src, snr_thresh=8.0):
#     thetaE_sub = SIS_thetaE_arcsec(M_sub)
#     x_sub, y_sub = pos_sub
#     x_src, y_src = pos_src

#     kwargs_sub = {'theta_E': thetaE_sub, 'center_x': x_sub, 'center_y': y_sub}
#     kwargs_lens       = [kwargs_main, kwargs_sub, kwargs_shear]
#     kwargs_lens_nosub = [kwargs_main,           kwargs_shear]

#     base = {'R_sersic': R_sersic, 'n_sersic': n_sersic, 'center_x': x_src, 'center_y': y_src}
#     src_g = [{**base, 'magnitude': mag_g}]
#     src_r = [{**base, 'magnitude': mag_g - g_r}]
#     src_i = [{**base, 'magnitude': mag_g - g_i}]
#     source_mags = [src_g, src_r, src_i]

#     # Semillas
#     seedA = int(rng.integers(0, 2**31-1))
#     seedB = seedA + 104729  # primo

#     # Por banda
#     imgs_clean_model, imgs_sub_model, imgs_src_model = [], [], []
#     imgs_clean_A, imgs_sub_A, imgs_clean_B = [], [], []
#     sig_clean, sig_sub = [], []

#     for b in range(3):
#         (m_clean, m_sub, m_src,
#          cA, sA, cB,
#          sC, sS) = cache.render_one_band(b, kwargs_lens, kwargs_lens_nosub, source_mags, seedA, seedB)
#         imgs_clean_model.append(m_clean); imgs_sub_model.append(m_sub); imgs_src_model.append(m_src)
#         imgs_clean_A.append(cA); imgs_sub_A.append(sA); imgs_clean_B.append(cB)
#         sig_clean.append(sC); sig_sub.append(sS)

#     # Stacks (H,W,3) en orden (g,r,i)
#     clean_model = np.stack(imgs_clean_model, axis=-1)
#     sub_model   = np.stack(imgs_sub_model,   axis=-1)
#     src_model   = np.stack(imgs_src_model,   axis=-1)
#     clean_A     = np.stack(imgs_clean_A,     axis=-1)
#     sub_A       = np.stack(imgs_sub_A,       axis=-1)
#     clean_B     = np.stack(imgs_clean_B,     axis=-1)
#     sigma_clean = np.stack(sig_clean,        axis=-1)
#     sigma_sub   = np.stack(sig_sub,          axis=-1)

#     # Residuales
#     resid_paired = (sub_A - clean_A).astype('f4')      # ruido pareado -> ~solo señal
#     resid_null   = (clean_B - clean_A).astype('f4')    # ruido independiente (nulo)
#     resid_diff   = (sub_A - clean_B).astype('f4')      # para χ² (indep.)

#     # SNR proxy (global, en toda la imagen)
#     R2_sig  = l2_rgb(resid_paired)
#     R2_null = l2_rgb(resid_null)
#     snr_proxy = rms_global(R2_sig) / (rms_global(R2_null) + 1e-12)

#     # χ² global coherente: Var = σ_sub^2 + σ_clean^2
#     var_diff = (sigma_sub**2 + sigma_clean**2).astype('f4')
#     chi2_map = (resid_diff**2) / np.clip(var_diff, 1e-12, None)
#     chi2_stat = float(np.sum(chi2_map))
#     dof = int(np.prod(chi2_map.shape))  # H*W*3
#     chi2_red = chi2_stat / max(dof, 1)

#     # Score 0-1 y etiqueta binaria (opcional)
#     score = float(chi2_to_score_sigmoid(chi2_red))
#     is_det = int(snr_proxy >= snr_thresh)

#     # Δψ (inspección)
#     x_flat, y_flat = cache.x_coords.ravel(), cache.y_coords.ravel()
#     psi_sub = cache.lm_psi_sub.potential(x_flat, y_flat, kwargs_lens      ).reshape(cache.x_coords.shape)
#     psi_cln = cache.lm_psi_clean.potential(x_flat, y_flat, kwargs_lens_nosub).reshape(cache.x_coords.shape)
#     delta_psi_map = (psi_sub - psi_cln).astype('f4')

#     # Visual source (composite RGB LSST-like)
#     source_rgb = sqrt_stretch_rgb(src_model)

#     # Salidas
#     return (sub_A.astype('f4'),            # images_rgb (sub + ruido)
#             clean_model.astype('f4'),      # images_clean (sin ruido)
#             delta_psi_map,                 # delta_psi_maps
#             resid_paired.astype('f4'),     # residual_rgb_sub (señal)
#             sigma_clean.astype('f4'), sigma_sub.astype('f4'),
#             float(snr_proxy), float(chi2_stat), int(dof), float(chi2_red),
#             score, int(is_det),
#             source_rgb.astype('f4'))

# # -----------------------------
# # Generación del dataset
# # -----------------------------
# if __name__ == "__main__":
#     # Config
#     N_TOTAL     = 300
#     SNR_THRESH  = 8.0
#     OUT_NAME    = 'LSST_chi2_lenstronomy.h5'

#     # Muestreo (masa log-uniforme)
#     mass_min, mass_max = 1e6, 1e10
#     logM_min, logM_max = np.log10(mass_min), np.log10(mass_max)
#     subhalo_pos_min, subhalo_pos_max = -1.6, 1.6
#     source_pos_min,  source_pos_max  = -1.0, 1.0

#     cache = RenderCache(use_reduced_shear=USE_REDUCED_SHEAR)

#     # Dummy para shapes
#     ex = simulate_pair_fast(cache, 1e8, (0.0,0.5), (0.0,0.0), snr_thresh=SNR_THRESH)
#     (ex_img, ex_clean, ex_dpsi, ex_rsig, ex_sigC, ex_sigS,
#      ex_snr, ex_c2, ex_dof, ex_c2r, ex_score, ex_det, ex_srcRGB) = ex
#     H, W, C = ex_img.shape

#     with h5py.File(OUT_NAME, 'w') as f:
#         d_img   = f.create_dataset('images_rgb',        (N_TOTAL, H, W, C), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         d_clean = f.create_dataset('images_clean',      (N_TOTAL, H, W, C), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         d_dpsi  = f.create_dataset('delta_psi_maps',    (N_TOTAL, H, W),    dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         d_rsig  = f.create_dataset('residual_rgb_sub',  (N_TOTAL, H, W, C), dtype='f4', chunks=True, compression='gzip', compression_opts=4)

#         d_sigC  = f.create_dataset('sigma_rgb_clean',   (N_TOTAL, H, W, C), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         d_sigS  = f.create_dataset('sigma_rgb_sub',     (N_TOTAL, H, W, C), dtype='f4', chunks=True, compression='gzip', compression_opts=4)

#         d_mass  = f.create_dataset('subhalo_mass',      (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         d_x     = f.create_dataset('subhalo_x',         (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         d_y     = f.create_dataset('subhalo_y',         (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         d_sx    = f.create_dataset('source_x',          (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         d_sy    = f.create_dataset('source_y',          (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)

#         d_snr   = f.create_dataset('snr_proxy',         (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         d_c2    = f.create_dataset('chi2_stat',         (N_TOTAL,), dtype='f8', chunks=True, compression='gzip', compression_opts=4)
#         d_dof   = f.create_dataset('chi2_dof',          (N_TOTAL,), dtype='i4', chunks=True, compression='gzip', compression_opts=4)
#         d_c2r   = f.create_dataset('chi2_reduced',      (N_TOTAL,), dtype='f8', chunks=True, compression='gzip', compression_opts=4)
#         d_score = f.create_dataset('detectability_score',(N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         d_det   = f.create_dataset('is_detectable',     (N_TOTAL,), dtype='i1', chunks=True, compression='gzip', compression_opts=4)

#         d_srcRGB = f.create_dataset('source_rgb',       (N_TOTAL, H, W, 3), dtype='f4', chunks=True, compression='gzip', compression_opts=4)

#         CHUNK, N_CH = 256, math.ceil(N_TOTAL/256)
#         for ch in range(N_CH):
#             a, b = ch*CHUNK, min(N_TOTAL, (ch+1)*CHUNK)
#             outs = []
#             for _ in range(b-a):
#                 M_sub = 10.0**rng.uniform(logM_min, logM_max)
#                 x_sub = rng.uniform(subhalo_pos_min, subhalo_pos_max)
#                 y_sub = rng.uniform(subhalo_pos_min, subhalo_pos_max)
#                 x_src = rng.uniform(source_pos_min,  source_pos_max)
#                 y_src = rng.uniform(source_pos_min,  source_pos_max)
#                 outs.append((simulate_pair_fast(cache, M_sub, (x_sub,y_sub), (x_src,y_src), snr_thresh=SNR_THRESH),
#                              (M_sub,x_sub,y_sub,x_src,y_src)))

#             for j, (ret, meta) in enumerate(outs):
#                 i = a + j
#                 (img, clean, dpsi, rsig, sigC, sigS,
#                  snr, c2, dof, c2r, score, det, srcRGB) = ret
#                 M_sub, x_sub, y_sub, x_src, y_src = meta

#                 d_img[i], d_clean[i], d_dpsi[i], d_rsig[i] = img, clean, dpsi, rsig
#                 d_sigC[i], d_sigS[i] = sigC, sigS

#                 d_mass[i], d_x[i], d_y[i], d_sx[i], d_sy[i] = M_sub, x_sub, y_sub, x_src, y_src
#                 d_snr[i], d_c2[i], d_dof[i], d_c2r[i], d_score[i], d_det[i] = snr, c2, dof, c2r, score, det
#                 d_srcRGB[i] = srcRGB

#             print(f"[chunk {ch+1}/{N_CH}] escrito: {a}–{b-1}")

#         # Atributos
#         f.attrs['N_samples']          = N_TOTAL
#         f.attrs['pixel_scale_arcsec'] = PIXEL_SCALE
#         f.attrs['stamp_size_arcsec']  = STAMP_SIZE_ARCSEC
#         f.attrs['bands']              = 'g,r,i'
#         f.attrs['psf']                = 'Gaussian, 10-year coadd'
#         f.attrs['z_lens_source']      = (float(z_lens), float(z_source))
#         f.attrs['H0_Om0_Ob0']         = (70.0, 0.3, 0.048)
#         f.attrs['snr_proxy_kind']     = 'paired residual (subA-cleanA) / null RMS (cleanB-cleanA)'
#         f.attrs['chi2_var']           = 'sigma_sub^2 + sigma_clean^2 (per-pixel, per-band)'
#         f.attrs['detectability_score']= f'logistic(a={SCORE_A}, b={SCORE_B}) over log10(chi2_red)'
#         f.attrs['shear_images']       = 'SHEAR_REDUCED' if USE_REDUCED_SHEAR else 'SHEAR'
#         f.attrs['shear_potential']    = 'SHEAR'
#         f.attrs['source_rgb_desc']    = 'LSST-like composite (R<-i,G<-r,B<-g) sqrt-stretch p95, in [0,1]'
# -*- coding: utf-8 -*-
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
from lenstronomy.Plots import plot_util  # para sqrt-stretch LSST-like

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

# Bandas LSST (coadd 10 años, PSF gaussiana)
LSST_g = LSST(band='g', psf_type='GAUSSIAN', coadd_years=10)
LSST_r = LSST(band='r', psf_type='GAUSSIAN', coadd_years=10)
LSST_i = LSST(band='i', psf_type='GAUSSIAN', coadd_years=10)
lsst_bands = [LSST_g, LSST_r, LSST_i]

# Fuente (Sérsic)
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
DO_SCALE = True                  # aplica sqrt-stretch a images_* guardadas (sólo visualización)
USE_REDUCED_SHEAR = True

# -----------------------------
# Utilidades
# -----------------------------
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

def rms_global(arr2):
    """RMS de un mapa 2D en toda la imagen."""
    return float(np.sqrt(np.mean(arr2*arr2)))

# --- Fuente “source-only” con PSF LSST y RGB “bonito” (para inspección)
def render_source_only_rgb(cache: "RenderCache", source_mags, add_noise=False, base_seed=0):
    imgs = []
    for b_idx, (sim, imSim, mags) in enumerate(zip(cache.sim_source, cache.im_source, source_mags)):
        _, kwargs_source, _ = sim.magnitude2amplitude([], mags, [])
        img = imSim.image([], kwargs_source, None, None)  # SIN lentes
        if add_noise:
            state = np.random.get_state()
            np.random.seed(int(base_seed) + b_idx)
            img = img + sim.noise_for_model(model=img)
            np.random.set_state(state)
        imgs.append(img.astype('f4'))
    src_lin = np.stack(imgs, axis=-1)  # (g,r,i)

    # sqrt-stretch por banda con p95 (LSST-like)
    def p95(a):
        flat = a.reshape(-1)
        if flat.size == 0:
            return 1.0
        k = int(0.95 * flat.size)
        k = np.clip(k, 0, flat.size - 1)
        return float(np.partition(flat, k)[k])

    g, r, i = src_lin[...,0], src_lin[...,1], src_lin[...,2]
    g95, r95, i95 = max(p95(g), 1e-8), max(p95(r), 1e-8), max(p95(i), 1e-8)

    src_rgb = np.zeros_like(src_lin)
    src_rgb[...,0] = plot_util.sqrt(i, scale_min=0, scale_max=i95)  # R <- i
    src_rgb[...,1] = plot_util.sqrt(r, scale_min=0, scale_max=r95)  # G <- r
    src_rgb[...,2] = plot_util.sqrt(g, scale_min=0, scale_max=g95)  # B <- g
    return src_rgb.astype('f4')

# -----------------------------
# RenderCache
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
        # pipeline solo-fuente (sin lente)
        self.kwargs_model_source = {
            'lens_model_list': [],
            'source_light_model_list': ['SERSIC'],
            'lens_light_model_list': [],
            'point_source_model_list': []
        }

        # configuración por banda
        self.band_configs = [LSST_g.kwargs_single_band(),
                             LSST_r.kwargs_single_band(),
                             LSST_i.kwargs_single_band()]
        self.sim_sub    = [SimAPI(numpix=NUMPIX, kwargs_single_band=bc, kwargs_model=self.kwargs_model_sub_img)
                           for bc in self.band_configs]
        self.sim_nosub  = [SimAPI(numpix=NUMPIX, kwargs_single_band=bc, kwargs_model=self.kwargs_model_nosub_img)
                           for bc in self.band_configs]
        self.sim_source = [SimAPI(numpix=NUMPIX, kwargs_single_band=bc, kwargs_model=self.kwargs_model_source)
                           for bc in self.band_configs]

        self.im_sub    = [sim.image_model_class(KWARGS_NUMERICS) for sim in self.sim_sub]
        self.im_nosub  = [sim.image_model_class(KWARGS_NUMERICS) for sim in self.sim_nosub]
        self.im_source = [sim.image_model_class(KWARGS_NUMERICS) for sim in self.sim_source]

        # grilla fija en plano de la imagen
        ra0 = -STAMP_SIZE_ARCSEC / 2.0
        dec0 = -STAMP_SIZE_ARCSEC / 2.0
        transform = np.eye(2) * PIXEL_SCALE
        pg = PixelGrid(nx=NUMPIX, ny=NUMPIX, ra_at_xy_0=ra0, dec_at_xy_0=dec0, transform_pix2angle=transform)
        self.x_coords, self.y_coords = pg.pixel_coordinates

        # LensModel para Δψ
        self.lm_psi_sub   = LensModel(lens_model_list=['EPL', 'SIS', 'SHEAR'])
        self.lm_psi_nosub = LensModel(lens_model_list=['EPL',        'SHEAR'])

# -----------------------------
# Núcleo: simulación + métricas
# -----------------------------
def paired_images_rgb(sim_list, im_list, kwargs_lens, source_mags, add_noise, base_seed):
    """Renderiza un stack (H,W,3) por banda, con o sin ruido, fijando semilla por banda."""
    imgs = []
    for b_idx, (sim, imSim, mags) in enumerate(zip(sim_list, im_list, source_mags)):
        _, kwargs_source, _ = sim.magnitude2amplitude([], mags, [])
        img = imSim.image(kwargs_lens, kwargs_source, None, None)
        if add_noise:
            state = np.random.get_state()
            np.random.seed(int(base_seed) + b_idx)
            img = img + sim.noise_for_model(model=img)
            np.random.set_state(state)
        imgs.append(img.astype('f4'))
    return np.stack(imgs, axis=-1)  # (H,W,3)

def simulate_pair_and_delta_fast(cache: "RenderCache",
                                 mass_subhalo, position_subhalo, source_position,
                                 snr_thresh_noisy=20.0):
    # ángulo de Einstein del subhalo (SIS)
    M_sub = mass_subhalo * u.M_sun
    thetaE_sub_rad = np.sqrt(4 * G * M_sub / c**2 * (D_ds / (D_d * D_s)))
    thetaE_sub = (thetaE_sub_rad * u.rad).to(u.arcsec).value

    x_sub, y_sub = position_subhalo
    kwargs_sub = {'theta_E': thetaE_sub, 'center_x': x_sub, 'center_y': y_sub}
    kwargs_lens       = [kwargs_main, kwargs_sub, kwargs_shear]
    kwargs_lens_nosub = [kwargs_main,           kwargs_shear]

    # fuente por banda (magnitudes)
    x_src, y_src = source_position
    base = {'R_sersic': R_sersic, 'n_sersic': n_sersic, 'center_x': x_src, 'center_y': y_src}
    src_g = [{**base, 'magnitude': mag_g}]
    src_r = [{**base, 'magnitude': mag_g - g_r}]
    src_i = [{**base, 'magnitude': mag_g - g_i}]
    source_mags = [src_g, src_r, src_i]

    # semillas (pareamos ruido por banda)
    seedA = int(rng.integers(0, 2**31 - 1))

    # cleanA: SIN subhalo y SIN ruido
    img_clean_A = paired_images_rgb(cache.sim_nosub, cache.im_nosub, kwargs_lens_nosub, source_mags,
                                    add_noise=False, base_seed=seedA)

    # subA: CON subhalo y CON ruido
    img_sub_A   = paired_images_rgb(cache.sim_sub,   cache.im_sub,   kwargs_lens,       source_mags,
                                    add_noise=True,  base_seed=seedA)

    # cleanB: SIN subhalo y CON ruido (ruido independiente)
    img_clean_B = paired_images_rgb(cache.sim_nosub, cache.im_nosub, kwargs_lens_nosub, source_mags,
                                    add_noise=True,  base_seed=seedA + 7919)

    # residuales (RGB, lineales)
    resid_sub_rgb  = (img_sub_A   - img_clean_A).astype('f4')   # señal+ruido
    resid_null_rgb = (img_clean_B - img_clean_A).astype('f4')   # solo ruido

    # ---------- SNR proxy (GLOBAL, sin máscara) ----------
    L2_sub  = l2_rgb(resid_sub_rgb)   # (H,W)
    L2_null = l2_rgb(resid_null_rgb)  # (H,W)
    rms_sub  = rms_global(L2_sub)
    rms_null = rms_global(L2_null)
    snr_eff  = rms_sub / (rms_null + 1e-12)
    is_det   = int(snr_eff >= snr_thresh_noisy)

    # ---------- χ² "signal" coherente con SNR ----------
    # Por diseño de este proxy: chi2_signal = SNR^2
    chi2_signal = float(snr_eff**2)

    # Δψ (mapa para inspección)
    x_flat, y_flat = cache.x_coords.ravel(), cache.y_coords.ravel()
    psi_sub   = cache.lm_psi_sub.potential(x_flat, y_flat, kwargs_lens).reshape(cache.x_coords.shape)
    psi_nosub = cache.lm_psi_nosub.potential(x_flat, y_flat, kwargs_lens_nosub).reshape(cache.x_coords.shape)
    delta_psi_map = (psi_sub - psi_nosub).astype('f4')

    # Fuente “source-only” como RGB LSST-like
    src_rgb = render_source_only_rgb(cache, source_mags, add_noise=False, base_seed=0)

    # Visual stretch para guardar imágenes “bonitas”
    if DO_SCALE:
        img_sub_vis,  _ , _ = sqrt_stretch_pair_fast(img_sub_A,           img_clean_A)          # noisy con sub
        _, img_clean_noiseless_vis, _ = sqrt_stretch_pair_fast(img_clean_A, img_clean_A)        # baseline clean (vis)
    else:
        img_sub_vis = img_sub_A
        img_clean_noiseless_vis = img_clean_A

    return (img_sub_vis.astype('f4'),
            img_clean_noiseless_vis.astype('f4'),
            delta_psi_map,
            resid_sub_rgb, resid_null_rgb,
            float(snr_eff), is_det,
            float(chi2_signal),
            src_rgb)

# -----------------------------
# Generación del dataset
# -----------------------------
if __name__ == "__main__":
    # Configuración dataset
    N_TOTAL     = 1000
    SNR_THRESH  = 20.0
    OUT_NAME    = 'LSST_chi2_dataset_2.h5'

    # Rangos de muestreo
    mass_min, mass_max = 1e6, 1e10
    subhalo_pos_min, subhalo_pos_max = -1.6, 1.6
    source_pos_min,  source_pos_max  = -1.0, 1.0

    # Cache
    cache = RenderCache(images_use_reduced_shear=USE_REDUCED_SHEAR)

    # Dummy para conocer shapes
    ex = simulate_pair_and_delta_fast(cache, 1e8, (0.2, 0.3), (0.0, 0.0),
                                      snr_thresh_noisy=SNR_THRESH)
    (ex_obs, ex_clean, ex_dpsi, ex_rsub, ex_rnull,
     ex_snr, ex_det, ex_chi2sig, ex_src_rgb) = ex
    ny, nx, n_channels = ex_obs.shape

    with h5py.File(OUT_NAME, 'w') as f:
        # Imágenes
        dset_input     = f.create_dataset('images_rgb',     (N_TOTAL, ny, nx, n_channels), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_clean     = f.create_dataset('images_clean',   (N_TOTAL, ny, nx, n_channels), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_delta_psi = f.create_dataset('delta_psi_maps', (N_TOTAL, ny, nx),             dtype='f4', chunks=True, compression='gzip', compression_opts=4)

        # Residuales RGB (lineales)
        dset_resid_sub  = f.create_dataset('residual_rgb_sub',  (N_TOTAL, ny, nx, 3), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_resid_null = f.create_dataset('residual_rgb_null', (N_TOTAL, ny, nx, 3), dtype='f4', chunks=True, compression='gzip', compression_opts=4)

        # Escalares/metadata por muestra
        dset_mass   = f.create_dataset('subhalo_mass', (N_TOTAL,), dtype='f8', chunks=True, compression='gzip', compression_opts=4)  # f8 para precisión
        dset_xpos   = f.create_dataset('subhalo_x',    (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_ypos   = f.create_dataset('subhalo_y',    (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_srcx   = f.create_dataset('source_x',     (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_srcy   = f.create_dataset('source_y',     (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)

        dset_snr    = f.create_dataset('snr_proxy',    (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
        dset_detect = f.create_dataset('is_detectable',(N_TOTAL,), dtype='i1', chunks=True, compression='gzip', compression_opts=4)

        # χ² derivada del SNR proxy (global)
        dset_chi2sig = f.create_dataset('chi2_signal', (N_TOTAL,), dtype='f8', chunks=True, compression='gzip', compression_opts=4)

        # Fuente (solo RGB LSST-like; mismo grid/PSF)
        dset_source_rgb = f.create_dataset('source_rgb', (N_TOTAL, ny, nx, 3), dtype='f4', chunks=True, compression='gzip', compression_opts=4)

        # Bucle (con chunks si quieres paralelizar)
        CHUNK  = 256
        N_CH   = math.ceil(N_TOTAL/CHUNK)
        N_JOBS = 0  # >0 para paralelo joblib

        def _simulate_one():
            M_sub = float(10**rng.uniform(np.log10(mass_min), np.log10(mass_max)))  # log-uniform por defecto
            x_sub = rng.uniform(subhalo_pos_min, subhalo_pos_max)
            y_sub = rng.uniform(subhalo_pos_min, subhalo_pos_max)
            x_src = rng.uniform(source_pos_min,  source_pos_max)
            y_src = rng.uniform(source_pos_min,  source_pos_max)
            return simulate_pair_and_delta_fast(cache, M_sub, (x_sub, y_sub), (x_src, y_src),
                                                snr_thresh_noisy=SNR_THRESH), (M_sub, x_sub, y_sub, x_src, y_src)

        for ch in range(N_CH):
            a = ch*CHUNK
            b = min(N_TOTAL, (ch+1)*CHUNK)
            n_this = b - a

            if N_JOBS and N_JOBS > 1:
                outs = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(_simulate_one)() for _ in range(n_this))
            else:
                outs = [_simulate_one() for _ in range(n_this)]

            for j, (ret, meta) in enumerate(outs):
                i = a + j
                (img_obs, img_clean, dpsi,
                 resid_sub_rgb, resid_null_rgb,
                 snr_eff, is_det, chi2sig, src_rgb) = ret
                M_sub, x_sub, y_sub, x_src, y_src = meta

                dset_input[i]     = img_obs
                dset_clean[i]     = img_clean
                dset_delta_psi[i] = dpsi
                dset_resid_sub[i]  = resid_sub_rgb
                dset_resid_null[i] = resid_null_rgb

                dset_mass[i] = M_sub
                dset_xpos[i] = x_sub
                dset_ypos[i] = y_sub
                dset_srcx[i] = x_src
                dset_srcy[i] = y_src

                dset_snr[i]     = snr_eff
                dset_detect[i]  = is_det
                dset_chi2sig[i] = chi2sig

                dset_source_rgb[i] = src_rgb

            print(f"[chunk {ch+1}/{N_CH}] escrito: {a}–{b-1}")

        # Atributos globales
        f.attrs['N_samples']               = N_TOTAL
        f.attrs['pixel_scale_arcsec']      = PIXEL_SCALE
        f.attrs['stamp_size_arcsec']       = STAMP_SIZE_ARCSEC
        f.attrs['bands']                   = 'g,r,i'
        f.attrs['psf']                     = 'Gaussian, 10-year coadd'
        f.attrs['images_clean_desc']       = 'sin subhalo y SIN ruido (visual: sqrt-stretch vs baseline)'
        f.attrs['images_input_desc']       = 'con subhalo y CON ruido (seed pareada por banda)'
        f.attrs['residual_maps_space']     = 'linear_RGB (residual_rgb_*)'
        f.attrs['shear_images']            = 'SHEAR_REDUCED' if USE_REDUCED_SHEAR else 'SHEAR'
        f.attrs['shear_potential']         = 'SHEAR'
        f.attrs['z_lens_source']           = (float(z_lens), float(z_source))
        f.attrs['H0_Om0_Ob0']              = (70.0, 0.3, 0.048)

        # Métricas (FULL FRAME)
        f.attrs['snr_proxy_kind']          = 'paired_noisy_residual_global (L2-RMS over FULL image)'
        f.attrs['snr_thresh_noisy']        = float(SNR_THRESH)
        f.attrs['chi2_signal_def']         = 'chi2_signal = (snr_proxy)^2 (global)'

        # Fuente
        f.attrs['source_rgb_desc']         = 'RGB LSST-like sqrt-stretch (R<-i, G<-r, B<-g) en [0,1], source-only; mismo grid y PSF'

    print(f"Archivo escrito: {OUT_NAME}")
