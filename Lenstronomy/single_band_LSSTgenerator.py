# -*- coding: utf-8 -*-
import numpy as np
import h5py
import math

from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import FlatLambdaCDM

from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util.param_util import phi_q2_ellipticity, shear_polar2cartesian

# ===========================================================
# CONFIGURACIÓN COSMOLÓGICA Y MODELOS BÁSICOS
# ===========================================================
z_lens, z_source = 0.881, 2.059
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048)
D_d  = cosmo.angular_diameter_distance(z_lens)
D_s  = cosmo.angular_diameter_distance(z_source)
D_ds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)

# Band LSST (usaremos solo g por ahora)
LSST_g = LSST(band='g', psf_type='GAUSSIAN', coadd_years=10)
kwargs_band = LSST_g.kwargs_single_band()

# Resolución de imagen
STAMP_SIZE_ARCSEC = 6.0
PIXEL_SCALE = kwargs_band['pixel_scale']
NUMPIX = int(round(STAMP_SIZE_ARCSEC / PIXEL_SCALE))
kwargs_numerics = {'point_source_supersampling_factor': 1}

# ===========================================================
# MODELOS Y PARAMETROS BASE
# ===========================================================
# Lente principal
theta_E_main, gamma_main = 1.452, 1.9
e1_main, e2_main = phi_q2_ellipticity(phi=np.deg2rad(-22.29), q=0.866)
kwargs_main = dict(theta_E=theta_E_main, gamma=gamma_main,
                   e1=e1_main, e2=e2_main, center_x=0.0, center_y=-0.1)
# Shear reducido
g1, g2 = shear_polar2cartesian(phi=np.deg2rad(107.9), gamma=0.015)
kwargs_shear = dict(gamma1=g1, gamma2=g2)

# Modelos
kwargs_model_sub = {
    'lens_model_list': ['EPL', 'SIS', 'SHEAR_REDUCED'],
    'source_light_model_list': ['SERSIC_ELLIPSE']
}
kwargs_model_nosub = {
    'lens_model_list': ['EPL', 'SHEAR_REDUCED'],
    'source_light_model_list': ['SERSIC_ELLIPSE']
}

# ===========================================================
# FUNCIONES AUXILIARES
# ===========================================================
def compute_thetaE_sub(mass_subhalo):
    """Einstein radius del subhalo en arcsec"""
    M_sub = mass_subhalo * u.M_sun
    thetaE_sub_rad = np.sqrt(4*G*M_sub/c**2 * (D_ds/(D_d*D_s)))
    return (thetaE_sub_rad * u.rad).to(u.arcsec).value

def compute_delta_psi(kwargs_lens_sub, kwargs_lens_nosub):
    """Mapa delta_psi = psi_sub - psi_nosub"""
    lm_sub = LensModel(lens_model_list=['EPL','SIS','SHEAR'])
    lm_nosub = LensModel(lens_model_list=['EPL','SHEAR'])
    x, y = np.meshgrid(np.linspace(-STAMP_SIZE_ARCSEC/2, STAMP_SIZE_ARCSEC/2, NUMPIX),
                       np.linspace(-STAMP_SIZE_ARCSEC/2, STAMP_SIZE_ARCSEC/2, NUMPIX))
    psi_sub = lm_sub.potential(x, y, kwargs_lens_sub)
    psi_nosub = lm_nosub.potential(x, y, kwargs_lens_nosub)
    return (psi_sub - psi_nosub).astype('f4')

def compute_chi2(image_sub, image_nosub_nonoise, sim):
    """ξ² y ξ²_red físicos"""
    noise_map = sim.estimate_noise(image_sub)
    xi_squared = np.sum(((image_sub - image_nosub_nonoise)**2) / (noise_map**2 + 1e-12))
    ndof = np.sum(np.isfinite(image_nosub_nonoise))
    xi_squared_red = xi_squared / ndof
    return float(xi_squared), float(xi_squared_red)

# ===========================================================
# FUNCIÓN PRINCIPAL DE SIMULACIÓN
# ===========================================================
def simulate_single_band(m_sub, pos_sub, pos_src):
    # Configurar simuladores
    sim_sub = SimAPI(numpix=NUMPIX, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model_sub)
    sim_nosub = SimAPI(numpix=NUMPIX, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model_nosub)

    # Fuente
    x_src, y_src = pos_src
    kwargs_source = [{
        'magnitude': 21,
        'R_sersic': 0.4,
        'n_sersic': 1,
        'e1': 0.05,
        'e2': -0.05,
        'center_x': x_src,
        'center_y': y_src
    }]
    _, kwargs_source_sub, _ = sim_sub.magnitude2amplitude([], kwargs_source, [])
    _, kwargs_source_nosub, _ = sim_nosub.magnitude2amplitude([], kwargs_source, [])

    # Subhalo
    thetaE_sub = compute_thetaE_sub(m_sub)
    kwargs_sub = {'theta_E': thetaE_sub, 'center_x': pos_sub[0], 'center_y': pos_sub[1]}

    # Combinar kwargs
    kwargs_lens_sub = [kwargs_main, kwargs_sub, kwargs_shear]
    kwargs_lens_nosub = [kwargs_main, kwargs_shear]

    # Crear imagen
    imSim_sub = sim_sub.image_model_class(kwargs_numerics)
    imSim_nosub = sim_nosub.image_model_class(kwargs_numerics)

    # Sin ruido (modelo base)
    image_nosub_nonoise = imSim_nosub.image(kwargs_lens_nosub, kwargs_source_nosub, None, None)

    # Con subhalo
    image_sub = imSim_sub.image(kwargs_lens_sub, kwargs_source_sub, None, None)

    # Agregar ruido
    image_sub += sim_sub.noise_for_model(image_sub)
    image_nosub = image_nosub_nonoise + sim_nosub.noise_for_model(image_nosub_nonoise)

    # Calcular métricas
    xi2, xi2_red = compute_chi2(image_sub, image_nosub_nonoise, sim_sub)
    delta_psi = compute_delta_psi(kwargs_lens_sub, kwargs_lens_nosub)

    return image_sub.astype('f4'), image_nosub_nonoise.astype('f4'), delta_psi, xi2, xi2_red

# ===========================================================
# GENERADOR DE DATASET
# ===========================================================
if __name__ == "__main__":
    N_TOTAL = 10000
    MASS_RANGE = (1e6, 1e9)
    SUB_POS_RANGE = (-1.6, 1.6)
    SRC_POS_RANGE = (-1.0, 1.0)
    OUT_NAME = "LSST_singleband_dataset.h5"

    with h5py.File(OUT_NAME, "w") as f:
        d_img_sub  = f.create_dataset("image_sub",  (N_TOTAL, NUMPIX, NUMPIX), dtype='f4', compression='gzip', compression_opts=4)
        d_img_nosub= f.create_dataset("image_nosub_nonoise", (N_TOTAL, NUMPIX, NUMPIX), dtype='f4', compression='gzip', compression_opts=4)
        d_dpsi     = f.create_dataset("delta_psi",  (N_TOTAL, NUMPIX, NUMPIX), dtype='f4', compression='gzip', compression_opts=4)
        d_xi2      = f.create_dataset("xi2",        (N_TOTAL,), dtype='f8', compression='gzip', compression_opts=4)
        d_xi2red   = f.create_dataset("xi2_reduced",(N_TOTAL,), dtype='f8', compression='gzip', compression_opts=4)
        d_mass     = f.create_dataset("subhalo_mass",(N_TOTAL,), dtype='f8', compression='gzip', compression_opts=4)
        d_xsub     = f.create_dataset("subhalo_x",(N_TOTAL,), dtype='f4', compression='gzip', compression_opts=4)
        d_ysub     = f.create_dataset("subhalo_y",(N_TOTAL,), dtype='f4', compression='gzip', compression_opts=4)
        d_xsrc     = f.create_dataset("source_x",(N_TOTAL,), dtype='f4', compression='gzip', compression_opts=4)
        d_ysrc     = f.create_dataset("source_y",(N_TOTAL,), dtype='f4', compression='gzip', compression_opts=4)

        for i in range(N_TOTAL):
            m_sub = 10**np.random.uniform(np.log10(MASS_RANGE[0]), np.log10(MASS_RANGE[1]))
            x_sub = np.random.uniform(SUB_POS_RANGE[0], SUB_POS_RANGE[1])
            y_sub = np.random.uniform(SUB_POS_RANGE[0], SUB_POS_RANGE[1])
            x_src = np.random.uniform(SRC_POS_RANGE[0], SRC_POS_RANGE[1])
            y_src = np.random.uniform(SRC_POS_RANGE[0], SRC_POS_RANGE[1])

            img_sub, img_nosub, dpsi, xi2, xi2red = simulate_single_band(m_sub, (x_sub,y_sub), (x_src,y_src))

            d_img_sub[i]   = img_sub
            d_img_nosub[i] = img_nosub
            d_dpsi[i]      = dpsi
            d_xi2[i]       = xi2
            d_xi2red[i]    = xi2red
            d_mass[i]      = m_sub
            d_xsub[i]      = x_sub
            d_ysub[i]      = y_sub
            d_xsrc[i]      = x_src
            d_ysrc[i]      = y_src

            if (i+1) % 50 == 0:
                print(f"[{i+1}/{N_TOTAL}] Done. Last χ²_red={xi2red:.3f}")

        f.attrs['description'] = "Single-band LSST-g lensing dataset with subhalo/no-subhalo pairs"
        f.attrs['num_samples'] = N_TOTAL
        f.attrs['mass_range']  = MASS_RANGE
        f.attrs['position_range'] = (SUB_POS_RANGE, SRC_POS_RANGE)
        f.attrs['pixel_scale'] = PIXEL_SCALE
        f.attrs['z_lens_source'] = (float(z_lens), float(z_source))
        f.attrs['band'] = 'LSST g'
        f.attrs['chi2_definition'] = "Σ (ΔI² / σ²), σ from estimate_noise"
        f.attrs['chi2_reduced'] = "χ² / Ndof"
        print(f"\nDataset saved as {OUT_NAME}")
