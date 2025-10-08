# # -*- coding: utf-8 -*-
# import math
# import numpy as np
# import h5py
# from joblib import Parallel, delayed

# from astropy import units as u
# from astropy.constants import G, c
# from astropy.cosmology import FlatLambdaCDM

# from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
# from lenstronomy.SimulationAPI.sim_api import SimAPI
# from lenstronomy.Data.pixel_grid import PixelGrid
# from lenstronomy.LensModel.lens_model import LensModel
# from lenstronomy.Util.param_util import phi_q2_ellipticity, shear_polar2cartesian
# from lenstronomy.Plots import plot_util  # para sqrt-stretch LSST-like

# # -----------------------------
# # Configuración global
# # -----------------------------
# SEED = 12345
# rng = np.random.default_rng(SEED)

# # Cosmología y distancias
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
# g_r = 1.0   # g - r
# g_i = 2.0   # g - i
# R_sersic = 0.15
# n_sersic = 1.0

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

# # Métrica/visual
# # Anillo más grueso (antes 0.30"). Recomendación típica: >= 1.5*FWHM_PSF o ~0.5–0.8"
# RING_HALFWIDTH = 0.60
# USE_REDUCED_SHEAR = True

# # Transformación a score continuo en [0,1] desde log10(chi2_red)
# SCORE_A = 2.0
# SCORE_B = 1.0

# # -----------------------------
# # Helpers
# # -----------------------------
# def ring_mask(x_coords, y_coords, x0, y0, theta_E, halfwidth=0.35):
#     r = np.hypot(x_coords - x0, y_coords - y0)
#     return (np.abs(r - theta_E) <= halfwidth)

# def paired_images_rgb(sim_list, im_list, kwargs_lens, source_mags, add_noise, base_seed):
#     """Renderiza un stack (H,W,3) por banda, con o sin ruido, fijando semilla por banda."""
#     imgs = []
#     for b_idx, (sim, imSim, mags) in enumerate(zip(sim_list, im_list, source_mags)):
#         _, kwargs_source, _ = sim.magnitude2amplitude([], mags, [])
#         img = imSim.image(kwargs_lens, kwargs_source, None, None)
#         if add_noise:
#             state = np.random.get_state()
#             np.random.seed(int(base_seed) + b_idx)
#             img = img + sim.noise_for_model(model=img)
#             np.random.set_state(state)
#         imgs.append(img.astype('f4'))
#     return np.stack(imgs, axis=-1)  # (H,W,3)

# def chi2_from_paired(resid_sub_rgb, cleanA_rgb, cleanB_rgb, mask=None, noiseless_A=True, eps_var=1e-12):
#     """
#     χ² con nulo emparejado.
#     - Si noiseless_A=True (cleanA SIN ruido): Var ≈ (cleanB - cleanA)^2
#     - Si noiseless_A=False (ambas ruidosas):  Var ≈ 0.5 * (cleanB - cleanA)^2
#     """
#     diff_null_rgb = cleanB_rgb - cleanA_rgb
#     var_rgb = (diff_null_rgb**2) if noiseless_A else 0.5 * (diff_null_rgb**2)
#     var_rgb = np.clip(var_rgb, eps_var, None)

#     w = resid_sub_rgb**2 / var_rgb
#     if mask is not None:
#         w = w[mask, :]
#     chi2 = float(np.sum(w))
#     dof  = int(w.size)
#     chi2_red = chi2 / max(dof, 1)
#     return chi2, dof, chi2_red

# def chi2_to_score_sigmoid(chi2_red, a=SCORE_A, b=SCORE_B):
#     # score ∈ [0,1] a partir de log10(chi2_red)
#     x = np.log10(np.clip(chi2_red, 1e-12, 1e20))
#     return 1.0 / (1.0 + np.exp(-a * (x - b)))

# # --- Fuente “source-only” con PSF LSST y RGB “bonito” (para inspección)
# def render_source_only_rgb(cache: "RenderCache", source_mags, add_noise=False, base_seed=0):
#     imgs = []
#     for b_idx, (sim, imSim, mags) in enumerate(zip(cache.sim_source, cache.im_source, source_mags)):
#         _, kwargs_source, _ = sim.magnitude2amplitude([], mags, [])
#         img = imSim.image([], kwargs_source, None, None)  # SIN lentes
#         if add_noise:
#             state = np.random.get_state()
#             np.random.seed(int(base_seed) + b_idx)
#             img = img + sim.noise_for_model(model=img)
#             np.random.set_state(state)
#         imgs.append(img.astype('f4'))
#     src_lin = np.stack(imgs, axis=-1)  # (g,r,i)

#     # sqrt-stretch por banda con p95 (LSST-like)
#     def p95(a):
#         flat = a.reshape(-1)
#         if flat.size == 0:
#             return 1.0
#         k = int(0.95 * flat.size)
#         k = np.clip(k, 0, flat.size - 1)
#         return float(np.partition(flat, k)[k])

#     g, r, i = src_lin[...,0], src_lin[...,1], src_lin[...,2]
#     g95, r95, i95 = max(p95(g), 1e-8), max(p95(r), 1e-8), max(p95(i), 1e-8)

#     src_rgb = np.zeros_like(src_lin)
#     src_rgb[...,0] = plot_util.sqrt(i, scale_min=0, scale_max=i95)  # R <- i
#     src_rgb[...,1] = plot_util.sqrt(r, scale_min=0, scale_max=r95)  # G <- r
#     src_rgb[...,2] = plot_util.sqrt(g, scale_min=0, scale_max=g95)  # B <- g
#     return src_rgb.astype('f4')

# # -----------------------------
# # RenderCache
# # -----------------------------
# class RenderCache:
#     def __init__(self, images_use_reduced_shear=True):
#         if images_use_reduced_shear:
#             self.lens_list_img_sub   = ['EPL', 'SIS', 'SHEAR_REDUCED']
#             self.lens_list_img_nosub = ['EPL',       'SHEAR_REDUCED']
#         else:
#             self.lens_list_img_sub   = ['EPL', 'SIS', 'SHEAR']
#             self.lens_list_img_nosub = ['EPL',       'SHEAR']

#         self.kwargs_model_sub_img = {
#             'lens_model_list': self.lens_list_img_sub,
#             'source_light_model_list': ['SERSIC'],
#             'lens_light_model_list': [],
#             'point_source_model_list': []
#         }
#         self.kwargs_model_nosub_img = {
#             'lens_model_list': self.lens_list_img_nosub,
#             'source_light_model_list': ['SERSIC'],
#             'lens_light_model_list': [],
#             'point_source_model_list': []
#         }
#         # pipeline solo-fuente (sin lente)
#         self.kwargs_model_source = {
#             'lens_model_list': [],
#             'source_light_model_list': ['SERSIC'],
#             'lens_light_model_list': [],
#             'point_source_model_list': []
#         }

#         # configuración por banda
#         self.band_configs = [LSST_g.kwargs_single_band(),
#                              LSST_r.kwargs_single_band(),
#                              LSST_i.kwargs_single_band()]
#         self.sim_sub    = [SimAPI(numpix=NUMPIX, kwargs_single_band=bc, kwargs_model=self.kwargs_model_sub_img)
#                            for bc in self.band_configs]
#         self.sim_nosub  = [SimAPI(numpix=NUMPIX, kwargs_single_band=bc, kwargs_model=self.kwargs_model_nosub_img)
#                            for bc in self.band_configs]
#         self.sim_source = [SimAPI(numpix=NUMPIX, kwargs_single_band=bc, kwargs_model=self.kwargs_model_source)
#                            for bc in self.band_configs]

#         self.im_sub    = [sim.image_model_class(KWARGS_NUMERICS) for sim in self.sim_sub]
#         self.im_nosub  = [sim.image_model_class(KWARGS_NUMERICS) for sim in self.sim_nosub]
#         self.im_source = [sim.image_model_class(KWARGS_NUMERICS) for sim in self.sim_source]

#         # grilla fija en plano de la imagen
#         ra0 = -STAMP_SIZE_ARCSEC / 2.0
#         dec0 = -STAMP_SIZE_ARCSEC / 2.0
#         transform = np.eye(2) * PIXEL_SCALE
#         pg = PixelGrid(nx=NUMPIX, ny=NUMPIX, ra_at_xy_0=ra0, dec_at_xy_0=dec0, transform_pix2angle=transform)
#         self.x_coords, self.y_coords = pg.pixel_coordinates

#         # máscara anular para la métrica (actualizada)
#         self.ring_mask = ring_mask(self.x_coords, self.y_coords,
#                                    kwargs_main['center_x'], kwargs_main['center_y'],
#                                    theta_E_main, halfwidth=RING_HALFWIDTH)

#         # modelos para potencial (Δψ)
#         self.lm_psi_sub   = LensModel(lens_model_list=['EPL', 'SIS', 'SHEAR'])
#         self.lm_psi_nosub = LensModel(lens_model_list=['EPL',        'SHEAR'])

# # -----------------------------
# # Núcleo: simulación + métricas
# # -----------------------------
# def simulate_pair_and_delta_fast(cache: "RenderCache",
#                                  mass_subhalo, position_subhalo, source_position,
#                                  snr_thresh_noisy=20.0):
#     # ángulo de Einstein del subhalo (SIS)
#     M_sub = mass_subhalo * u.M_sun
#     thetaE_sub_rad = np.sqrt(4 * G * M_sub / c**2 * (D_ds / (D_d * D_s)))
#     thetaE_sub = (thetaE_sub_rad * u.rad).to(u.arcsec).value

#     x_sub, y_sub = position_subhalo
#     kwargs_sub = {'theta_E': thetaE_sub, 'center_x': x_sub, 'center_y': y_sub}
#     kwargs_lens       = [kwargs_main, kwargs_sub, kwargs_shear]
#     kwargs_lens_nosub = [kwargs_main,           kwargs_shear]

#     # fuente por banda (magnitudes)
#     x_src, y_src = source_position
#     base = {'R_sersic': R_sersic, 'n_sersic': n_sersic, 'center_x': x_src, 'center_y': y_src}
#     src_g = [{**base, 'magnitude': mag_g}]
#     src_r = [{**base, 'magnitude': mag_g - g_r}]
#     src_i = [{**base, 'magnitude': mag_g - g_i}]
#     source_mags = [src_g, src_r, src_i]

#     # semillas (pareamos ruido por banda cuando corresponda)
#     seedA = int(rng.integers(0, 2**31 - 1))

#     # cleanA: SIN subhalo y SIN ruido
#     img_clean_A = paired_images_rgb(cache.sim_nosub, cache.im_nosub, kwargs_lens_nosub, source_mags,
#                                     add_noise=False, base_seed=seedA)

#     # subA: CON subhalo y CON ruido
#     img_sub_A   = paired_images_rgb(cache.sim_sub,   cache.im_sub,   kwargs_lens,       source_mags,
#                                     add_noise=True,  base_seed=seedA)

#     # cleanB: SIN subhalo y CON ruido (para estimar Var del ruido)
#     img_clean_B = paired_images_rgb(cache.sim_nosub, cache.im_nosub, kwargs_lens_nosub, source_mags,
#                                     add_noise=True,  base_seed=seedA + 7919)

#     # residuales (RGB, lineales)
#     resid_sub_rgb  = (img_sub_A   - img_clean_A).astype('f4')   # señal+ruido
#     resid_null_rgb = (img_clean_B - img_clean_A).astype('f4')   # solo ruido

#     # SNR proxy (rms de L2 en máscara)
#     L2_sub  = np.sqrt(np.sum(resid_sub_rgb**2, axis=-1))
#     L2_null = np.sqrt(np.sum(resid_null_rgb**2, axis=-1))
#     if cache.ring_mask.any():
#         rms_sub  = float(np.sqrt(np.mean(L2_sub[cache.ring_mask]**2)))
#         rms_null = float(np.sqrt(np.mean(L2_null[cache.ring_mask]**2)))
#     else:
#         rms_sub  = float(np.sqrt(np.mean(L2_sub**2)))
#         rms_null = float(np.sqrt(np.mean(L2_null**2)))
#     snr_eff = rms_sub / (rms_null + 1e-12)

#     # χ² emparejado (A es noiseless)
#     chi2, dof, chi2_red = chi2_from_paired(resid_sub_rgb, img_clean_A, img_clean_B,
#                                            mask=cache.ring_mask, noiseless_A=True)

#     # score continuo en [0,1] a partir de chi2_red
#     detect_score = float(chi2_to_score_sigmoid(chi2_red, a=SCORE_A, b=SCORE_B))

#     # Δψ (mapa, opcional para inspección)
#     x_flat, y_flat = cache.x_coords.ravel(), cache.y_coords.ravel()
#     psi_sub   = cache.lm_psi_sub.potential(x_flat, y_flat, kwargs_lens).reshape(cache.x_coords.shape)
#     psi_nosub = cache.lm_psi_nosub.potential(x_flat, y_flat, kwargs_lens_nosub).reshape(cache.x_coords.shape)
#     delta_psi_map = (psi_sub - psi_nosub).astype('f4')

#     # Fuente “source-only” como RGB LSST-like
#     src_rgb = render_source_only_rgb(cache, source_mags, add_noise=False, base_seed=0)

#     is_det = int(snr_eff >= snr_thresh_noisy)  # etiqueta binaria (método viejo)

#     # Salidas:
#     return (img_sub_A.astype('f4'), img_clean_A.astype('f4'), delta_psi_map,
#             resid_sub_rgb, resid_null_rgb,
#             float(snr_eff), is_det,
#             float(chi2), int(dof), float(chi2_red),
#             detect_score,  # <--- NUEVO
#             src_rgb)

# # -----------------------------
# # Generación del dataset
# # -----------------------------
# if __name__ == "__main__":
#     # Configuración dataset
#     N_TOTAL     = 1000
#     SNR_THRESH  = 20.0
#     OUT_NAME    = 'LSST_chi2_dataset.h5'

#     # Rangos de muestreo
#     mass_min, mass_max = 1e6, 1e10
#     subhalo_pos_min, subhalo_pos_max = -1.6, 1.6
#     source_pos_min,  source_pos_max  = -1.0, 1.0

#     # Cache
#     cache = RenderCache(images_use_reduced_shear=USE_REDUCED_SHEAR)

#     # Dummy para conocer shapes
#     ex = simulate_pair_and_delta_fast(cache, 1e8, (0.2, 0.3), (0.0, 0.0),
#                                       snr_thresh_noisy=SNR_THRESH)
#     (ex_obs, ex_clean, ex_dpsi, ex_rsub, ex_rnull,
#      ex_snr, ex_det, ex_chi2, ex_dof, ex_chi2r,
#      ex_score, ex_src_rgb) = ex
#     ny, nx, n_channels = ex_obs.shape

#     with h5py.File(OUT_NAME, 'w') as f:
#         # Imágenes
#         dset_input     = f.create_dataset('images_rgb',     (N_TOTAL, ny, nx, n_channels), dtype='f4', chunks=True, compression='gzip', compression_opts=4)  # subhalo+ruido
#         dset_clean     = f.create_dataset('images_clean',   (N_TOTAL, ny, nx, n_channels), dtype='f4', chunks=True, compression='gzip', compression_opts=4)  # sin subhalo, SIN ruido
#         dset_delta_psi = f.create_dataset('delta_psi_maps', (N_TOTAL, ny, nx),             dtype='f4', chunks=True, compression='gzip', compression_opts=4)

#         # Residuales RGB (lineales)
#         dset_resid_sub  = f.create_dataset('residual_rgb_sub',  (N_TOTAL, ny, nx, 3), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         dset_resid_null = f.create_dataset('residual_rgb_null', (N_TOTAL, ny, nx, 3), dtype='f4', chunks=True, compression='gzip', compression_opts=4)

#         # Escalares/metadata por muestra
#         dset_mass   = f.create_dataset('subhalo_mass', (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         dset_xpos   = f.create_dataset('subhalo_x',    (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         dset_ypos   = f.create_dataset('subhalo_y',    (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         dset_srcx   = f.create_dataset('source_x',     (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         dset_srcy   = f.create_dataset('source_y',     (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)

#         dset_snr    = f.create_dataset('snr_proxy',    (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)
#         dset_detect = f.create_dataset('is_detectable',(N_TOTAL,), dtype='i1', chunks=True, compression='gzip', compression_opts=4)

#         # χ² (y score)
#         dset_chi2    = f.create_dataset('chi2_stat',    (N_TOTAL,), dtype='f8', chunks=True, compression='gzip', compression_opts=4)
#         dset_chi2dof = f.create_dataset('chi2_dof',     (N_TOTAL,), dtype='i4', chunks=True, compression='gzip', compression_opts=4)
#         dset_chi2red = f.create_dataset('chi2_reduced', (N_TOTAL,), dtype='f8', chunks=True, compression='gzip', compression_opts=4)
#         dset_score   = f.create_dataset('detectability_score', (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)  # <--- NUEVO

#         # Fuente (solo RGB LSST-like; sin lineal y sin 2D)
#         dset_source_rgb = f.create_dataset('source_rgb', (N_TOTAL, ny, nx, 3), dtype='f4', chunks=True, compression='gzip', compression_opts=4)

#         # Bucle (con chunks si quieres paralelizar)
#         CHUNK  = 256
#         N_CH   = math.ceil(N_TOTAL/CHUNK)
#         N_JOBS = 0  # >0 para paralelo joblib

#         def _simulate_one():
#             M_sub = rng.uniform(mass_min, mass_max)
#             x_sub = rng.uniform(subhalo_pos_min, subhalo_pos_max)
#             y_sub = rng.uniform(subhalo_pos_min, subhalo_pos_max)
#             x_src = rng.uniform(source_pos_min,  source_pos_max)
#             y_src = rng.uniform(source_pos_min,  source_pos_max)
#             return simulate_pair_and_delta_fast(cache, M_sub, (x_sub, y_sub), (x_src, y_src),
#                                                 snr_thresh_noisy=SNR_THRESH), (M_sub, x_sub, y_sub, x_src, y_src)

#         for ch in range(N_CH):
#             a = ch*CHUNK
#             b = min(N_TOTAL, (ch+1)*CHUNK)
#             n_this = b - a

#             if N_JOBS and N_JOBS > 1:
#                 outs = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(_simulate_one)() for _ in range(n_this))
#             else:
#                 outs = [_simulate_one() for _ in range(n_this)]

#             for j, (ret, meta) in enumerate(outs):
#                 i = a + j
#                 (img_obs, img_clean, dpsi,
#                  resid_sub_rgb, resid_null_rgb,
#                  snr_eff, is_det,
#                  chi2, dof, chi2_red,
#                  detect_score, src_rgb) = ret
#                 M_sub, x_sub, y_sub, x_src, y_src = meta

#                 dset_input[i]     = img_obs
#                 dset_clean[i]     = img_clean
#                 dset_delta_psi[i] = dpsi
#                 dset_resid_sub[i]  = resid_sub_rgb
#                 dset_resid_null[i] = resid_null_rgb

#                 dset_mass[i] = M_sub
#                 dset_xpos[i] = x_sub
#                 dset_ypos[i] = y_sub
#                 dset_srcx[i] = x_src
#                 dset_srcy[i] = y_src

#                 dset_snr[i]    = snr_eff
#                 dset_detect[i] = is_det

#                 dset_chi2[i]    = chi2
#                 dset_chi2dof[i] = dof
#                 dset_chi2red[i] = chi2_red
#                 dset_score[i]   = detect_score     # <--- NUEVO

#                 dset_source_rgb[i] = src_rgb

#             print(f"[chunk {ch+1}/{N_CH}] escrito: {a}–{b-1}")

#         # Atributos globales
#         f.attrs['N_samples']               = N_TOTAL
#         f.attrs['pixel_scale_arcsec']      = PIXEL_SCALE
#         f.attrs['stamp_size_arcsec']       = STAMP_SIZE_ARCSEC
#         f.attrs['bands']                   = 'g,r,i'
#         f.attrs['psf']                     = 'Gaussian, 10-year coadd'
#         f.attrs['images_clean_desc']       = 'sin subhalo y SIN ruido'
#         f.attrs['images_input_desc']       = 'con subhalo y CON ruido (seed pareada por banda)'
#         f.attrs['residual_maps_space']     = 'linear_RGB'
#         f.attrs['shear_images']            = 'SHEAR_REDUCED' if USE_REDUCED_SHEAR else 'SHEAR'
#         f.attrs['shear_potential']         = 'SHEAR'
#         f.attrs['z_lens_source']           = (float(z_lens), float(z_source))
#         f.attrs['H0_Om0_Ob0']              = (70.0, 0.3, 0.048)
#         f.attrs['ring_halfwidth']          = RING_HALFWIDTH
#         f.attrs['theta_E_main']            = float(theta_E_main)

#         f.attrs['snr_proxy_kind']          = 'paired_noisy_residual_global (L2-RMS on ring mask)'
#         f.attrs['snr_thresh_noisy']        = float(SNR_THRESH)

#         f.attrs['detectability_criterion'] = 'paired_chi2_over_null en máscara (per-pixel, per-band)'
#         f.attrs['chi2_notes']              = ('cleanA SIN ruido; Var ≈ (cleanB-cleanA)^2 por pixel y banda; '
#                                               'chi2=sum(resid_sub^2/Var) en máscara; dof=Npix_masked*3')
#         f.attrs['detectability_score_desc'] = ('sigmoid(a*(log10(chi2_reduced)-b)) con a={:.3g}, b={:.3g}; '
#                                                'score ∈ [0,1]'.format(SCORE_A, SCORE_B))

#         f.attrs['source_rgb_desc']         = 'RGB LSST-like sqrt-stretch (R<-i, G<-r, B<-g) en [0,1], source-only; mismo grid y PSF'

#     print(f"Archivo escrito: {OUT_NAME}")

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

# Métrica/visual
RING_HALFWIDTH = 0.60   # anillo más grueso
USE_REDUCED_SHEAR = True

# Transformación a score continuo en [0,1] desde log10(chi2_red)
SCORE_A = 2.0
SCORE_B = 1.0

# Muestreo de masas
MASS_SAMPLING_MODE = "loguniform"   # opciones: "loguniform" | "powerlaw" | "uniform"
MASS_POWERLAW_ALPHA = 1.9           # sólo si mode="powerlaw"

# -----------------------------
# Helpers
# -----------------------------
def ring_mask(x_coords, y_coords, x0, y0, theta_E, halfwidth=0.35):
    r = np.hypot(x_coords - x0, y_coords - y0)
    return (np.abs(r - theta_E) <= halfwidth)

def sample_subhalo_mass(rng, m_min, m_max, mode="loguniform", alpha=1.9):
    """
    Devuelve una masa M en [m_min, m_max] según:
      - "loguniform": uniforme en log10 M
      - "powerlaw":   dN/dM ∝ M^{-alpha}
      - "uniform":    uniforme en M (no recomendado)
    """
    if mode == "loguniform":
        logM = rng.uniform(np.log10(m_min), np.log10(m_max))
        return float(10.0**logM)
    elif mode == "uniform":
        return float(rng.uniform(m_min, m_max))
    elif mode == "powerlaw":
        # Inversa de CDF para alpha != 1
        if abs(alpha - 1.0) < 1e-8:  # límite -> log-uniforme
            u = rng.random()
            return float(m_min * (m_max/m_min)**u)
        a = 1.0 - alpha
        u = rng.random()
        return float((u*(m_max**a - m_min**a) + m_min**a)**(1.0/a))
    else:
        raise ValueError(f"mode desconocido: {mode}")

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

def chi2_from_paired(resid_sub_rgb, cleanA_rgb, cleanB_rgb, mask=None, noiseless_A=True, eps_var=1e-12):
    """
    χ² con nulo emparejado.
    - Si noiseless_A=True (cleanA SIN ruido): Var ≈ (cleanB - cleanA)^2
    - Si noiseless_A=False (ambas ruidosas):  Var ≈ 0.5 * (cleanB - cleanA)^2
    """
    diff_null_rgb = cleanB_rgb - cleanA_rgb
    var_rgb = (diff_null_rgb**2) if noiseless_A else 0.5 * (diff_null_rgb**2)
    var_rgb = np.clip(var_rgb, eps_var, None)

    w = resid_sub_rgb**2 / var_rgb
    if mask is not None:
        w = w[mask, :]
    chi2 = float(np.sum(w))
    dof  = int(w.size)
    chi2_red = chi2 / max(dof, 1)
    return chi2, dof, chi2_red

def chi2_to_score_sigmoid(chi2_red, a=SCORE_A, b=SCORE_B):
    # score ∈ [0,1] a partir de log10(chi2_red)
    x = np.log10(np.clip(chi2_red, 1e-12, 1e20))
    return 1.0 / (1.0 + np.exp(-a * (x - b)))

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

        # máscara anular para la métrica (actualizada)
        self.ring_mask = ring_mask(self.x_coords, self.y_coords,
                                   kwargs_main['center_x'], kwargs_main['center_y'],
                                   theta_E_main, halfwidth=RING_HALFWIDTH)

        # modelos para potencial (Δψ)
        self.lm_psi_sub   = LensModel(lens_model_list=['EPL', 'SIS', 'SHEAR'])
        self.lm_psi_nosub = LensModel(lens_model_list=['EPL',        'SHEAR'])

# -----------------------------
# Núcleo: simulación + métricas
# -----------------------------
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

    # semillas (pareamos ruido por banda cuando corresponda)
    seedA = int(rng.integers(0, 2**31 - 1))

    # cleanA: SIN subhalo y SIN ruido
    img_clean_A = paired_images_rgb(cache.sim_nosub, cache.im_nosub, kwargs_lens_nosub, source_mags,
                                    add_noise=False, base_seed=seedA)

    # subA: CON subhalo y CON ruido
    img_sub_A   = paired_images_rgb(cache.sim_sub,   cache.im_sub,   kwargs_lens,       source_mags,
                                    add_noise=True,  base_seed=seedA)

    # cleanB: SIN subhalo y CON ruido (para estimar Var del ruido)
    img_clean_B = paired_images_rgb(cache.sim_nosub, cache.im_nosub, kwargs_lens_nosub, source_mags,
                                    add_noise=True,  base_seed=seedA + 7919)

    # residuales (RGB, lineales)
    resid_sub_rgb  = (img_sub_A   - img_clean_A).astype('f4')   # señal+ruido
    resid_null_rgb = (img_clean_B - img_clean_A).astype('f4')   # solo ruido

    # SNR proxy (rms de L2 en máscara)
    L2_sub  = np.sqrt(np.sum(resid_sub_rgb**2, axis=-1))
    L2_null = np.sqrt(np.sum(resid_null_rgb**2, axis=-1))
    if cache.ring_mask.any():
        rms_sub  = float(np.sqrt(np.mean(L2_sub[cache.ring_mask]**2)))
        rms_null = float(np.sqrt(np.mean(L2_null[cache.ring_mask]**2)))
    else:
        rms_sub  = float(np.sqrt(np.mean(L2_sub**2)))
        rms_null = float(np.sqrt(np.mean(L2_null**2)))
    snr_eff = rms_sub / (rms_null + 1e-12)

    # χ² emparejado (A es noiseless)
    chi2, dof, chi2_red = chi2_from_paired(resid_sub_rgb, img_clean_A, img_clean_B,
                                           mask=cache.ring_mask, noiseless_A=True)

    # score continuo en [0,1] a partir de chi2_red
    detect_score = float(chi2_to_score_sigmoid(chi2_red, a=SCORE_A, b=SCORE_B))

    # Δψ (mapa, opcional para inspección)
    x_flat, y_flat = cache.x_coords.ravel(), cache.y_coords.ravel()
    psi_sub   = cache.lm_psi_sub.potential(x_flat, y_flat, kwargs_lens).reshape(cache.x_coords.shape)
    psi_nosub = cache.lm_psi_nosub.potential(x_flat, y_flat, kwargs_lens_nosub).reshape(cache.x_coords.shape)
    delta_psi_map = (psi_sub - psi_nosub).astype('f4')

    # Fuente “source-only” como RGB LSST-like
    src_rgb = render_source_only_rgb(cache, source_mags, add_noise=False, base_seed=0)

    is_det = int(snr_eff >= snr_thresh_noisy)  # etiqueta binaria (método viejo)

    # Salidas:
    return (img_sub_A.astype('f4'), img_clean_A.astype('f4'), delta_psi_map,
            resid_sub_rgb, resid_null_rgb,
            float(snr_eff), is_det,
            float(chi2), int(dof), float(chi2_red),
            detect_score,
            src_rgb)

# -----------------------------
# Generación del dataset
# -----------------------------
if __name__ == "__main__":
    # Configuración dataset
    N_TOTAL     = 1000
    SNR_THRESH  = 20.0
    OUT_NAME    = 'LSST_chi2_dataset.h5'

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
     ex_snr, ex_det, ex_chi2, ex_dof, ex_chi2r,
     ex_score, ex_src_rgb) = ex
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

        # χ² (y score)
        dset_chi2    = f.create_dataset('chi2_stat',    (N_TOTAL,), dtype='f8', chunks=True, compression='gzip', compression_opts=4)
        dset_chi2dof = f.create_dataset('chi2_dof',     (N_TOTAL,), dtype='i4', chunks=True, compression='gzip', compression_opts=4)
        dset_chi2red = f.create_dataset('chi2_reduced', (N_TOTAL,), dtype='f8', chunks=True, compression='gzip', compression_opts=4)
        dset_score   = f.create_dataset('detectability_score', (N_TOTAL,), dtype='f4', chunks=True, compression='gzip', compression_opts=4)

        # Fuente (solo RGB LSST-like)
        dset_source_rgb = f.create_dataset('source_rgb', (N_TOTAL, ny, nx, 3), dtype='f4', chunks=True, compression='gzip', compression_opts=4)

        # Bucle (con chunks si quieres paralelizar)
        CHUNK  = 256
        N_CH   = math.ceil(N_TOTAL/CHUNK)
        N_JOBS = 0  # >0 para paralelo joblib

        def _simulate_one():
            # --- muestreo corregido de masa ---
            M_sub = sample_subhalo_mass(rng, mass_min, mass_max,
                                        mode=MASS_SAMPLING_MODE, alpha=MASS_POWERLAW_ALPHA)
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
                 snr_eff, is_det,
                 chi2, dof, chi2_red,
                 detect_score, src_rgb) = ret
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

                dset_snr[i]    = snr_eff
                dset_detect[i] = is_det

                dset_chi2[i]    = chi2
                dset_chi2dof[i] = dof
                dset_chi2red[i] = chi2_red
                dset_score[i]   = detect_score

                dset_source_rgb[i] = src_rgb

            print(f"[chunk {ch+1}/{N_CH}] escrito: {a}–{b-1}")

        # Atributos globales
        f.attrs['N_samples']               = N_TOTAL
        f.attrs['pixel_scale_arcsec']      = PIXEL_SCALE
        f.attrs['stamp_size_arcsec']       = STAMP_SIZE_ARCSEC
        f.attrs['bands']                   = 'g,r,i'
        f.attrs['psf']                     = 'Gaussian, 10-year coadd'
        f.attrs['images_clean_desc']       = 'sin subhalo y SIN ruido'
        f.attrs['images_input_desc']       = 'con subhalo y CON ruido (seed pareada por banda)'
        f.attrs['residual_maps_space']     = 'linear_RGB'
        f.attrs['shear_images']            = 'SHEAR_REDUCED' if USE_REDUCED_SHEAR else 'SHEAR'
        f.attrs['shear_potential']         = 'SHEAR'
        f.attrs['z_lens_source']           = (float(z_lens), float(z_source))
        f.attrs['H0_Om0_Ob0']              = (70.0, 0.3, 0.048)
        f.attrs['ring_halfwidth']          = RING_HALFWIDTH
        f.attrs['theta_E_main']            = float(theta_E_main)

        # Muestreo de masas (documentación)
        f.attrs['subhalo_mass_sampling'] = MASS_SAMPLING_MODE
        f.attrs['subhalo_mass_range']    = (float(mass_min), float(mass_max))
        f.attrs['subhalo_mass_alpha']    = float(MASS_POWERLAW_ALPHA)

        # Métricas
        f.attrs['snr_proxy_kind']          = 'paired_noisy_residual_global (L2-RMS on ring mask)'
        f.attrs['snr_thresh_noisy']        = float(SNR_THRESH)
        f.attrs['detectability_criterion'] = 'paired_chi2_over_null en máscara (per-pixel, per-band)'
        f.attrs['chi2_notes']              = ('cleanA SIN ruido; Var ≈ (cleanB-cleanA)^2 por pixel y banda; '
                                              'chi2=sum(resid_sub^2/Var) en máscara; dof=Npix_masked*3')
        f.attrs['detectability_score_desc'] = ('sigmoid(a*(log10(chi2_reduced)-b)) con a={:.3g}, b={:.3g}; '
                                               'score ∈ [0,1]'.format(SCORE_A, SCORE_B))
        f.attrs['source_rgb_desc']         = 'RGB LSST-like sqrt-stretch (R<-i, G<-r, B<-g) en [0,1], source-only; mismo grid y PSF'

    print(f"Archivo escrito: {OUT_NAME}")
