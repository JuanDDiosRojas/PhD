# -*- coding: utf-8 -*-
import numpy as np
import h5py

from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import FlatLambdaCDM

from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util.param_util import phi_q2_ellipticity, shear_polar2cartesian

# ===========================================================
# COSMOLOGY & INSTRUMENT
# ===========================================================
z_lens, z_source = 0.881, 2.059
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048)
D_d  = cosmo.angular_diameter_distance(z_lens)
D_s  = cosmo.angular_diameter_distance(z_source)
D_ds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)

# LSST single band (g)
LSST_g = LSST(band='g', psf_type='GAUSSIAN', coadd_years=10)
kwargs_band = LSST_g.kwargs_single_band()

STAMP_SIZE_ARCSEC = 6.0
PIXEL_SCALE = kwargs_band['pixel_scale']
NUMPIX = int(round(STAMP_SIZE_ARCSEC / PIXEL_SCALE))
kwargs_numerics = {'point_source_supersampling_factor': 1}

# ===========================================================
# LENS MODELS (base)
# ===========================================================
theta_E_main, gamma_main = 1.452, 1.9
e1_main, e2_main = phi_q2_ellipticity(phi=np.deg2rad(-22.29), q=0.866)
kwargs_main = dict(theta_E=theta_E_main, gamma=gamma_main,
                   e1=e1_main, e2=e2_main, center_x=0.0, center_y=-0.1)

g1, g2 = shear_polar2cartesian(phi=np.deg2rad(107.9), gamma=0.015)
kwargs_shear = dict(gamma1=g1, gamma2=g2)

kwargs_model_sub = {
    'lens_model_list': ['EPL', 'SIS', 'SHEAR_REDUCED'],
    'source_light_model_list': ['SERSIC_ELLIPSE']
}
kwargs_model_nosub = {
    'lens_model_list': ['EPL', 'SHEAR_REDUCED'],
    'source_light_model_list': ['SERSIC_ELLIPSE']
}

# ===========================================================
# BIC SETTINGS (adjust if you change parameterization)
# ===========================================================
# k_sub: number of free params in (EPL + SIS + SHEAR_REDUCED) model
# k_nosub: number of free params in (EPL + SHEAR_REDUCED) model
K_SUB   = 7   # your suggested value
K_NOSUB = 4   # your suggested value

# ===========================================================
# HELPERS
# ===========================================================
def compute_thetaE_sub(mass_subhalo):
    """Einstein radius of subhalo in arcsec."""
    M_sub = mass_subhalo * u.M_sun
    thetaE_sub_rad = np.sqrt(4*G*M_sub/c**2 * (D_ds/(D_d*D_s)))
    return (thetaE_sub_rad * u.rad).to(u.arcsec).value

def compute_delta_psi(kwargs_lens_sub, kwargs_lens_nosub):
    """Delta potential map: psi_sub - psi_nosub on the same grid."""
    lm_sub = LensModel(lens_model_list=['EPL','SIS','SHEAR'])
    lm_nosub = LensModel(lens_model_list=['EPL','SHEAR'])
    x, y = np.meshgrid(np.linspace(-STAMP_SIZE_ARCSEC/2, STAMP_SIZE_ARCSEC/2, NUMPIX),
                       np.linspace(-STAMP_SIZE_ARCSEC/2, STAMP_SIZE_ARCSEC/2, NUMPIX))
    psi_sub   = lm_sub.potential(x, y, kwargs_lens_sub)
    psi_nosub = lm_nosub.potential(x, y, kwargs_lens_nosub)
    return (psi_sub - psi_nosub).astype('f4')

def chi2_vs_model(image_obs, model_nonoise, sim):
    """
    Proper chi^2 for Gaussian noise: sum ( (obs - model)^2 / sigma^2 ).
    sigma map is estimated from the observed image (same instrument setup).
    """
    sigma = sim.estimate_noise(image_obs)
    num = (image_obs - model_nonoise)**2
    den = sigma**2 + 1e-12
    chi2 = float(np.sum(num / den))
    n = int(np.sum(np.isfinite(model_nonoise)))  # number of valid pixels
    chi2_red = chi2 / max(n, 1)
    return chi2, chi2_red, n

def bic_from_chi2(chi2, k, n):
    """BIC = chi2 + k*ln(n) under Gaussian likelihood approximation."""
    return chi2 + k * np.log(max(n, 1))

# ===========================================================
# MAIN SIMULATION (single band)
# ===========================================================
def simulate_single_band(m_sub, pos_sub, pos_src):
    # Simulators for both hypotheses
    sim_sub   = SimAPI(numpix=NUMPIX, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model_sub)
    sim_nosub = SimAPI(numpix=NUMPIX, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model_nosub)

    # Source (same for both)
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
    _, kwargs_source_sub, _   = sim_sub.magnitude2amplitude([], kwargs_source, [])
    _, kwargs_source_nosub, _ = sim_nosub.magnitude2amplitude([], kwargs_source, [])

    # Subhalo (SIS) for the "sub" model
    thetaE_sub = compute_thetaE_sub(m_sub)
    kwargs_sub = {'theta_E': thetaE_sub, 'center_x': pos_sub[0], 'center_y': pos_sub[1]}

    # Combine lens kwargs
    kwargs_lens_sub   = [kwargs_main, kwargs_sub, kwargs_shear]
    kwargs_lens_nosub = [kwargs_main,            kwargs_shear]

    # Image renderers
    imSim_sub   = sim_sub.image_model_class(kwargs_numerics)
    imSim_nosub = sim_nosub.image_model_class(kwargs_numerics)

    # --- noiseless model images (theoretical predictions) ---
    model_sub_nonoise   = imSim_sub.image(  kwargs_lens_sub,   kwargs_source_sub,   None, None)
    model_nosub_nonoise = imSim_nosub.image(kwargs_lens_nosub, kwargs_source_nosub, None, None)

    # --- observed noisy image (we assume the real sky is the "sub" world) ---
    image_obs = model_sub_nonoise + sim_sub.noise_for_model(model_sub_nonoise)

    # --- chi^2 against each model (using the SAME observed image) ---
    chi2_sub,   chi2r_sub,   n_sub   = chi2_vs_model(image_obs, model_sub_nonoise,   sim_sub)
    chi2_nosub, chi2r_nosub, n_nosub = chi2_vs_model(image_obs, model_nosub_nonoise, sim_nosub)

    # --- BICs ---
    n_pix = min(n_sub, n_nosub)
    BIC_sub   = bic_from_chi2(chi2_sub,   K_SUB,   n_pix)
    BIC_nosub = bic_from_chi2(chi2_nosub, K_NOSUB, n_pix)
    dBIC = BIC_nosub - BIC_sub   # > 0 favors "sub" model

    # --- delta-psi map ---
    delta_psi = compute_delta_psi(kwargs_lens_sub, kwargs_lens_nosub)

    # --- source-only image (unlensed), with/without noise) ---
    # Construimos un SimAPI sin lente (lista vacía) y convertimos magnitudes en ese contexto
    kwargs_model_source = {
        'lens_model_list': [],
        'source_light_model_list': ['SERSIC_ELLIPSE']
    }
    sim_source = SimAPI(numpix=NUMPIX, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model_source)
    _, kwargs_source_only, _ = sim_source.magnitude2amplitude([], kwargs_source, [])
    imSim_source = sim_source.image_model_class(kwargs_numerics)

    # Nota: pasar [] (no None) para "no lens"
    source_only_nonoise = imSim_source.image([], kwargs_source_only, None, None)
    source_only_noisy   = source_only_nonoise + sim_source.noise_for_model(source_only_nonoise)

    return {
        "image_obs":            image_obs.astype('f4'),
        "model_sub_nonoise":    model_sub_nonoise.astype('f4'),
        "model_nosub_nonoise":  model_nosub_nonoise.astype('f4'),
        "delta_psi":            delta_psi.astype('f4'),
        "chi2_sub":   float(chi2_sub),   "chi2r_sub":   float(chi2r_sub),
        "chi2_nosub": float(chi2_nosub), "chi2r_nosub": float(chi2r_nosub),
        "BIC_sub":    float(BIC_sub),    "BIC_nosub":   float(BIC_nosub),
        "dBIC":       float(dBIC),
        "source_only_nonoise":  source_only_nonoise.astype('f4'),
        "source_only_noisy":    source_only_noisy.astype('f4'),
        "n_pix":      int(n_pix)
    }


# ===========================================================
# DATASET WRITER
# ===========================================================
if __name__ == "__main__":
    N_TOTAL = 50000
    MASS_RANGE   = (1e6, 1e9)
    SUB_POS_RANGE = (-1.6, 1.6)
    SRC_POS_RANGE = (-1.0, 1.0)
    OUT_NAME = "LSST_sb_dataset_BIC.h5"

    with h5py.File(OUT_NAME, "w") as f:
        # Core images
        d_img_obs   = f.create_dataset("image_sub",            (N_TOTAL, NUMPIX, NUMPIX), dtype='f4', compression='gzip', compression_opts=4)  # observed noisy (SUB world)
        d_sub_clean = f.create_dataset("image_sub_nonoise",    (N_TOTAL, NUMPIX, NUMPIX), dtype='f4', compression='gzip', compression_opts=4)  # model (SUB) noiseless
        d_nosub_clean = f.create_dataset("image_nosub_nonoise",(N_TOTAL, NUMPIX, NUMPIX), dtype='f4', compression='gzip', compression_opts=4)  # model (NO-SUB) noiseless
        d_dpsi      = f.create_dataset("delta_psi",            (N_TOTAL, NUMPIX, NUMPIX), dtype='f4', compression='gzip', compression_opts=4)

        # Source-only (unlensed) images
        d_src_clean = f.create_dataset("source_only_nonoise",  (N_TOTAL, NUMPIX, NUMPIX), dtype='f4', compression='gzip', compression_opts=4)
        d_src_noisy = f.create_dataset("source_only_noisy",    (N_TOTAL, NUMPIX, NUMPIX), dtype='f4', compression='gzip', compression_opts=4)

        # Scalars: chi2, chi2_red, BICs
        d_chi2_sub    = f.create_dataset("chi2_sub",    (N_TOTAL,), dtype='f8', compression='gzip', compression_opts=4)
        d_chi2r_sub   = f.create_dataset("chi2r_sub",   (N_TOTAL,), dtype='f8', compression='gzip', compression_opts=4)
        d_chi2_nosub  = f.create_dataset("chi2_nosub",  (N_TOTAL,), dtype='f8', compression='gzip', compression_opts=4)
        d_chi2r_nosub = f.create_dataset("chi2r_nosub", (N_TOTAL,), dtype='f8', compression='gzip', compression_opts=4)
        d_BIC_sub     = f.create_dataset("BIC_sub",     (N_TOTAL,), dtype='f8', compression='gzip', compression_opts=4)
        d_BIC_nosub   = f.create_dataset("BIC_nosub",   (N_TOTAL,), dtype='f8', compression='gzip', compression_opts=4)
        d_dBIC        = f.create_dataset("dBIC",        (N_TOTAL,), dtype='f8', compression='gzip', compression_opts=4)
        d_npix        = f.create_dataset("n_pix",       (N_TOTAL,), dtype='i8', compression='gzip', compression_opts=4)

        # Physics: mass/geometry
        d_mass = f.create_dataset("subhalo_mass", (N_TOTAL,), dtype='f8', compression='gzip', compression_opts=4)
        d_xsub = f.create_dataset("subhalo_x",    (N_TOTAL,), dtype='f4', compression='gzip', compression_opts=4)
        d_ysub = f.create_dataset("subhalo_y",    (N_TOTAL,), dtype='f4', compression='gzip', compression_opts=4)
        d_xsrc = f.create_dataset("source_x",     (N_TOTAL,), dtype='f4', compression='gzip', compression_opts=4)
        d_ysrc = f.create_dataset("source_y",     (N_TOTAL,), dtype='f4', compression='gzip', compression_opts=4)

        for i in range(N_TOTAL):
            m_sub = 10**np.random.uniform(np.log10(MASS_RANGE[0]), np.log10(MASS_RANGE[1]))
            x_sub = np.random.uniform(SUB_POS_RANGE[0], SUB_POS_RANGE[1])
            y_sub = np.random.uniform(SUB_POS_RANGE[0], SUB_POS_RANGE[1])
            x_src = np.random.uniform(SRC_POS_RANGE[0], SRC_POS_RANGE[1])
            y_src = np.random.uniform(SRC_POS_RANGE[0], SRC_POS_RANGE[1])

            out = simulate_single_band(m_sub, (x_sub, y_sub), (x_src, y_src))

            # Write images
            d_img_obs[i]    = out["image_obs"]
            d_sub_clean[i]  = out["model_sub_nonoise"]
            d_nosub_clean[i]= out["model_nosub_nonoise"]
            d_dpsi[i]       = out["delta_psi"]
            d_src_clean[i]  = out["source_only_nonoise"]
            d_src_noisy[i]  = out["source_only_noisy"]

            # Metrics
            d_chi2_sub[i]    = out["chi2_sub"]
            d_chi2r_sub[i]   = out["chi2r_sub"]
            d_chi2_nosub[i]  = out["chi2_nosub"]
            d_chi2r_nosub[i] = out["chi2r_nosub"]
            d_BIC_sub[i]     = out["BIC_sub"]
            d_BIC_nosub[i]   = out["BIC_nosub"]
            d_dBIC[i]        = out["dBIC"]
            d_npix[i]        = out["n_pix"]

            # Physics
            d_mass[i] = m_sub
            d_xsub[i] = x_sub
            d_ysub[i] = y_sub
            d_xsrc[i] = x_src
            d_ysrc[i] = y_src

            if (i+1) % 50 == 0:
                print(f"[{i+1}/{N_TOTAL}] χ²_red(sub)={out['chi2r_sub']:.3f} | ΔBIC={out['dBIC']:.2f}")

        # file attrs
        f.attrs['description'] = "Single-band LSST-g simulated lensing dataset (SUB world obs) with BIC/chi2 for sub vs nosub models."
        f.attrs['num_samples'] = N_TOTAL
        f.attrs['mass_range']  = MASS_RANGE
        f.attrs['position_range'] = (SUB_POS_RANGE, SRC_POS_RANGE)
        f.attrs['pixel_scale'] = PIXEL_SCALE
        f.attrs['z_lens_source'] = (float(z_lens), float(z_source))
        f.attrs['band'] = 'LSST g'
        f.attrs['bic_definition'] = "BIC = chi2 + k ln(n); chi2 computed vs noiseless model with Gaussian sigma from estimate_noise(image_obs)"
        f.attrs['k_sub'] = K_SUB
        f.attrs['k_nosub'] = K_NOSUB
        f.attrs['delta_bic'] = "dBIC = BIC_sub - BIC_nosub (negative values favor SUB model)"
        print(f"\nDataset saved as {OUT_NAME}")
