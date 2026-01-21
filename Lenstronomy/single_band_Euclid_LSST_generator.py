# -*- coding: utf-8 -*-
import numpy as np
import h5py

from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import FlatLambdaCDM

from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util.param_util import phi_q2_ellipticity, shear_polar2cartesian

# ===========================================================
# USER CONFIG
# ===========================================================
INSTRUMENT   = "EUCLID"   # "LSST" or "EUCLID"
LSST_BAND    = "g"
EUCLID_BAND  = "VIS"
COADD_LSST_YEARS  = 10
COADD_EUCLID_YEARS = 6

# What to compute (toggle what you need)
COMPUTE_CHI2   = True     # to get chi2_sub/chi2_nosub (+ reduced versions)
COMPUTE_DELTA  = True     # to get dchi2 = chi2_nosub - chi2_sub (and reduced)
COMPUTE_BIC    = False    # this time you said you don't need BIC

# What to save (images)
SAVE_IMAGE_OBS         = True
SAVE_SUB_NOISELESS     = True
SAVE_SUB_NOISY         = True
SAVE_NOSUB_NOISELESS   = True
SAVE_NOSUB_NOISY       = True
SAVE_DELTA_PSI         = True
SAVE_SOURCE_ONLY       = True

# ===========================================================
# COSMOLOGY
# ===========================================================
z_lens, z_source = 0.881, 2.059
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048)
D_d  = cosmo.angular_diameter_distance(z_lens)
D_s  = cosmo.angular_diameter_distance(z_source)
D_ds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)

# ===========================================================
# INSTRUMENT CONFIG
# ===========================================================
def make_single_band_config():
    inst_name = INSTRUMENT.upper()
    if inst_name == "LSST":
        inst = LSST(band=LSST_BAND, psf_type="GAUSSIAN", coadd_years=COADD_LSST_YEARS)
        band_label = f"LSST-{LSST_BAND}"
    elif inst_name == "EUCLID":
        inst = Euclid(band=EUCLID_BAND, coadd_years=COADD_EUCLID_YEARS)
        band_label = f"EUCLID-{EUCLID_BAND}"
    else:
        raise ValueError(f"INSTRUMENT must be 'LSST' or 'EUCLID', got '{INSTRUMENT}'.")
    return inst.kwargs_single_band(), band_label

kwargs_band, BAND_LABEL = make_single_band_config()

STAMP_SIZE_ARCSEC = 6.464
PIXEL_SCALE = float(kwargs_band["pixel_scale"])
NUMPIX = int(round(STAMP_SIZE_ARCSEC / PIXEL_SCALE))
kwargs_numerics = {"point_source_supersampling_factor": 1}

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
    "lens_model_list": ["EPL", "SIS", "SHEAR_REDUCED"],
    "source_light_model_list": ["SERSIC_ELLIPSE"],
}
kwargs_model_nosub = {
    "lens_model_list": ["EPL", "SHEAR_REDUCED"],
    "source_light_model_list": ["SERSIC_ELLIPSE"],
}

# for BIC if you ever enable it again
K_SUB   = 7
K_NOSUB = 4

# ===========================================================
# HELPERS
# ===========================================================
def compute_thetaE_sub(mass_subhalo):
    """Einstein radius of subhalo in arcsec."""
    M_sub = mass_subhalo * u.M_sun
    thetaE_sub_rad = np.sqrt(4 * G * M_sub / c**2 * (D_ds / (D_d * D_s)))
    return (thetaE_sub_rad * u.rad).to(u.arcsec).value

def compute_delta_psi(kwargs_lens_sub, kwargs_lens_nosub):
    """Delta potential map: psi_sub - psi_nosub on the same grid."""
    lm_sub   = LensModel(lens_model_list=["EPL", "SIS", "SHEAR"])
    lm_nosub = LensModel(lens_model_list=["EPL", "SHEAR"])
    grid = np.linspace(-STAMP_SIZE_ARCSEC/2, STAMP_SIZE_ARCSEC/2, NUMPIX)
    x, y = np.meshgrid(grid, grid)
    psi_sub   = lm_sub.potential(x, y, kwargs_lens_sub)
    psi_nosub = lm_nosub.potential(x, y, kwargs_lens_nosub)
    return (psi_sub - psi_nosub).astype("f4")

def chi2_vs_model(image_obs, model_nonoise, sim):
    """
    chi^2 = sum ( (obs - model)^2 / sigma^2 )
    sigma map estimated from observed image using sim.estimate_noise.
    """
    sigma = sim.estimate_noise(image_obs)
    num = (image_obs - model_nonoise) ** 2
    den = sigma ** 2 + 1e-12
    chi2 = float(np.sum(num / den))
    n = int(np.sum(np.isfinite(model_nonoise)))
    chi2_red = chi2 / max(n, 1)
    return chi2, chi2_red, n

def bic_from_chi2(chi2, k, n):
    return chi2 + k * np.log(max(n, 1))

# ===========================================================
# MAIN SIMULATION
# ===========================================================
def simulate_single_band(m_sub, pos_sub, pos_src):
    # Build simulators (same band, different lens model lists)
    sim_sub   = SimAPI(numpix=NUMPIX, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model_sub)
    sim_nosub = SimAPI(numpix=NUMPIX, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model_nosub)

    # Source
    x_src, y_src = pos_src
    kwargs_source = [{
        "magnitude": 21,
        "R_sersic": 0.4,
        "n_sersic": 1,
        "e1": 0.05,
        "e2": -0.05,
        "center_x": x_src,
        "center_y": y_src,
    }]
    _, kwargs_source_sub, _   = sim_sub.magnitude2amplitude([], kwargs_source, [])
    _, kwargs_source_nosub, _ = sim_nosub.magnitude2amplitude([], kwargs_source, [])

    # Subhalo
    thetaE_sub = compute_thetaE_sub(m_sub)
    kwargs_sub = {"theta_E": thetaE_sub, "center_x": pos_sub[0], "center_y": pos_sub[1]}

    kwargs_lens_sub   = [kwargs_main, kwargs_sub, kwargs_shear]
    kwargs_lens_nosub = [kwargs_main,            kwargs_shear]

    imSim_sub   = sim_sub.image_model_class(kwargs_numerics)
    imSim_nosub = sim_nosub.image_model_class(kwargs_numerics)

    # noiseless models
    model_sub_nonoise   = imSim_sub.image(  kwargs_lens_sub,   kwargs_source_sub,   None, None)
    model_nosub_nonoise = imSim_nosub.image(kwargs_lens_nosub, kwargs_source_nosub, None, None)

    # noisy versions (optional saved products)
    model_sub_noisy   = model_sub_nonoise   + sim_sub.noise_for_model(model_sub_nonoise)
    model_nosub_noisy = model_nosub_nonoise + sim_nosub.noise_for_model(model_nosub_nonoise)

    # observed image: assume sky is SUB world
    image_obs = model_sub_nonoise + sim_sub.noise_for_model(model_sub_nonoise)

    # delta-psi
    delta_psi = compute_delta_psi(kwargs_lens_sub, kwargs_lens_nosub) if SAVE_DELTA_PSI else None

    # source-only
    if SAVE_SOURCE_ONLY:
        kwargs_model_source = {"lens_model_list": [], "source_light_model_list": ["SERSIC_ELLIPSE"]}
        sim_source = SimAPI(numpix=NUMPIX, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model_source)
        _, kwargs_source_only, _ = sim_source.magnitude2amplitude([], kwargs_source, [])
        imSim_source = sim_source.image_model_class(kwargs_numerics)
        source_only_nonoise = imSim_source.image([], kwargs_source_only, None, None)
        source_only_noisy   = source_only_nonoise + sim_source.noise_for_model(source_only_nonoise)
    else:
        source_only_nonoise = source_only_noisy = None

    # metrics
    chi2_sub = chi2r_sub = chi2_nosub = chi2r_nosub = np.nan
    dchi2 = dchi2r = np.nan
    BIC_sub = BIC_nosub = dBIC = np.nan
    n_pix = int(np.sum(np.isfinite(model_sub_nonoise)))

    if COMPUTE_CHI2 or COMPUTE_BIC or COMPUTE_DELTA:
        chi2_sub,   chi2r_sub,   n_sub   = chi2_vs_model(image_obs, model_sub_nonoise,   sim_sub)
        chi2_nosub, chi2r_nosub, n_nosub = chi2_vs_model(image_obs, model_nosub_nonoise, sim_nosub)
        n_pix = min(n_sub, n_nosub)

    if COMPUTE_DELTA:
        dchi2  = float(chi2_nosub - chi2_sub)
        dchi2r = float(chi2r_nosub - chi2r_sub)

    if COMPUTE_BIC:
        BIC_sub   = float(bic_from_chi2(chi2_sub,   K_SUB,   n_pix))
        BIC_nosub = float(bic_from_chi2(chi2_nosub, K_NOSUB, n_pix))
        dBIC      = float(BIC_nosub - BIC_sub)

    return {
        # images
        "image_obs":            image_obs.astype("f4"),
        "model_sub_nonoise":    model_sub_nonoise.astype("f4"),
        "model_sub_noisy":      model_sub_noisy.astype("f4"),
        "model_nosub_nonoise":  model_nosub_nonoise.astype("f4"),
        "model_nosub_noisy":    model_nosub_noisy.astype("f4"),
        "delta_psi":            None if delta_psi is None else delta_psi.astype("f4"),
        "source_only_nonoise":  None if source_only_nonoise is None else source_only_nonoise.astype("f4"),
        "source_only_noisy":    None if source_only_noisy   is None else source_only_noisy.astype("f4"),

        # scalars
        "chi2_sub":   float(chi2_sub),
        "chi2r_sub":  float(chi2r_sub),
        "chi2_nosub": float(chi2_nosub),
        "chi2r_nosub":float(chi2r_nosub),
        "dchi2":      float(dchi2),
        "dchi2r":     float(dchi2r),
        "BIC_sub":    float(BIC_sub),
        "BIC_nosub":  float(BIC_nosub),
        "dBIC":       float(dBIC),
        "n_pix":      int(n_pix),
    }

# ===========================================================
# DATASET WRITER
# ===========================================================
if __name__ == "__main__":
    N_TOTAL = 50000
    MASS_RANGE    = (1e6, 1e9)
    SUB_POS_RANGE = (-1.6, 1.6)
    SRC_POS_RANGE = (-1.0, 1.0)

    tag_parts = [INSTRUMENT.upper(), BAND_LABEL, "singleband"]
    if COMPUTE_DELTA: tag_parts.append("dCHI2")
    if COMPUTE_BIC:   tag_parts.append("BIC")
    OUT_NAME = "_".join(tag_parts) + ".h5"

    with h5py.File(OUT_NAME, "w") as f:
        # ---- Images ----
        if SAVE_IMAGE_OBS:
            d_img_obs = f.create_dataset("image_obs", (N_TOTAL, NUMPIX, NUMPIX),
                                         dtype="f4", compression="gzip", compression_opts=4)

        if SAVE_SUB_NOISELESS:
            d_sub_clean = f.create_dataset("image_sub_nonoise", (N_TOTAL, NUMPIX, NUMPIX),
                                           dtype="f4", compression="gzip", compression_opts=4)

        if SAVE_SUB_NOISY:
            d_sub_noisy = f.create_dataset("image_sub_noisy", (N_TOTAL, NUMPIX, NUMPIX),
                                           dtype="f4", compression="gzip", compression_opts=4)

        if SAVE_NOSUB_NOISELESS:
            d_nosub_clean = f.create_dataset("image_nosub_nonoise", (N_TOTAL, NUMPIX, NUMPIX),
                                             dtype="f4", compression="gzip", compression_opts=4)

        if SAVE_NOSUB_NOISY:
            d_nosub_noisy = f.create_dataset("image_nosub_noisy", (N_TOTAL, NUMPIX, NUMPIX),
                                             dtype="f4", compression="gzip", compression_opts=4)

        if SAVE_DELTA_PSI:
            d_dpsi = f.create_dataset("delta_psi", (N_TOTAL, NUMPIX, NUMPIX),
                                      dtype="f4", compression="gzip", compression_opts=4)

        if SAVE_SOURCE_ONLY:
            d_src_clean = f.create_dataset("source_only_nonoise", (N_TOTAL, NUMPIX, NUMPIX),
                                           dtype="f4", compression="gzip", compression_opts=4)
            d_src_noisy = f.create_dataset("source_only_noisy", (N_TOTAL, NUMPIX, NUMPIX),
                                           dtype="f4", compression="gzip", compression_opts=4)

        # ---- Scalars ----
        if COMPUTE_CHI2 or COMPUTE_DELTA or COMPUTE_BIC:
            d_chi2_sub    = f.create_dataset("chi2_sub",    (N_TOTAL,), dtype="f8", compression="gzip", compression_opts=4)
            d_chi2r_sub   = f.create_dataset("chi2r_sub",   (N_TOTAL,), dtype="f8", compression="gzip", compression_opts=4)
            d_chi2_nosub  = f.create_dataset("chi2_nosub",  (N_TOTAL,), dtype="f8", compression="gzip", compression_opts=4)
            d_chi2r_nosub = f.create_dataset("chi2r_nosub", (N_TOTAL,), dtype="f8", compression="gzip", compression_opts=4)

        if COMPUTE_DELTA:
            d_dchi2  = f.create_dataset("dchi2",  (N_TOTAL,), dtype="f8", compression="gzip", compression_opts=4)
            d_dchi2r = f.create_dataset("dchi2r", (N_TOTAL,), dtype="f8", compression="gzip", compression_opts=4)

        if COMPUTE_BIC:
            d_BIC_sub   = f.create_dataset("BIC_sub",   (N_TOTAL,), dtype="f8", compression="gzip", compression_opts=4)
            d_BIC_nosub = f.create_dataset("BIC_nosub", (N_TOTAL,), dtype="f8", compression="gzip", compression_opts=4)
            d_dBIC      = f.create_dataset("dBIC",      (N_TOTAL,), dtype="f8", compression="gzip", compression_opts=4)

        d_npix = f.create_dataset("n_pix", (N_TOTAL,), dtype="i8", compression="gzip", compression_opts=4)

        # ---- Physics ----
        d_mass = f.create_dataset("subhalo_mass", (N_TOTAL,), dtype="f8", compression="gzip", compression_opts=4)
        d_xsub = f.create_dataset("subhalo_x",    (N_TOTAL,), dtype="f4", compression="gzip", compression_opts=4)
        d_ysub = f.create_dataset("subhalo_y",    (N_TOTAL,), dtype="f4", compression="gzip", compression_opts=4)
        d_xsrc = f.create_dataset("source_x",     (N_TOTAL,), dtype="f4", compression="gzip", compression_opts=4)
        d_ysrc = f.create_dataset("source_y",     (N_TOTAL,), dtype="f4", compression="gzip", compression_opts=4)

        # ---- Loop ----
        for i in range(N_TOTAL):
            m_sub = 10**np.random.uniform(np.log10(MASS_RANGE[0]), np.log10(MASS_RANGE[1]))
            x_sub = np.random.uniform(*SUB_POS_RANGE)
            y_sub = np.random.uniform(*SUB_POS_RANGE)
            x_src = np.random.uniform(*SRC_POS_RANGE)
            y_src = np.random.uniform(*SRC_POS_RANGE)

            out = simulate_single_band(m_sub, (x_sub, y_sub), (x_src, y_src))

            # images
            if SAVE_IMAGE_OBS:       d_img_obs[i]     = out["image_obs"]
            if SAVE_SUB_NOISELESS:   d_sub_clean[i]   = out["model_sub_nonoise"]
            if SAVE_SUB_NOISY:       d_sub_noisy[i]   = out["model_sub_noisy"]
            if SAVE_NOSUB_NOISELESS: d_nosub_clean[i] = out["model_nosub_nonoise"]
            if SAVE_NOSUB_NOISY:     d_nosub_noisy[i] = out["model_nosub_noisy"]
            if SAVE_DELTA_PSI:       d_dpsi[i]        = out["delta_psi"]
            if SAVE_SOURCE_ONLY:
                d_src_clean[i] = out["source_only_nonoise"]
                d_src_noisy[i] = out["source_only_noisy"]

            # scalars
            if COMPUTE_CHI2 or COMPUTE_DELTA or COMPUTE_BIC:
                d_chi2_sub[i]    = out["chi2_sub"]
                d_chi2r_sub[i]   = out["chi2r_sub"]
                d_chi2_nosub[i]  = out["chi2_nosub"]
                d_chi2r_nosub[i] = out["chi2r_nosub"]

            if COMPUTE_DELTA:
                d_dchi2[i]  = out["dchi2"]
                d_dchi2r[i] = out["dchi2r"]

            if COMPUTE_BIC:
                d_BIC_sub[i]   = out["BIC_sub"]
                d_BIC_nosub[i] = out["BIC_nosub"]
                d_dBIC[i]      = out["dBIC"]

            d_npix[i] = out["n_pix"]

            # physics
            d_mass[i] = m_sub
            d_xsub[i] = x_sub
            d_ysub[i] = y_sub
            d_xsrc[i] = x_src
            d_ysrc[i] = y_src

            if (i + 1) % 50 == 0:
                msg = f"[{i+1}/{N_TOTAL}]"
                if COMPUTE_DELTA:
                    msg += f"  Δχ²={out['dchi2']:.2f}  Δχ²_red={out['dchi2r']:.4f}"
                elif COMPUTE_CHI2:
                    msg += f"  χ²_red(sub)={out['chi2r_sub']:.3f}"
                print(msg)

        # ---- Attributes ----
        f.attrs["description"] = f"Single-band lensing dataset ({BAND_LABEL}) with configurable image products and metrics."
        f.attrs["instrument"] = INSTRUMENT.upper()
        f.attrs["band"] = BAND_LABEL
        f.attrs["num_samples"] = int(N_TOTAL)
        f.attrs["stamp_size_arcsec"] = float(STAMP_SIZE_ARCSEC)
        f.attrs["pixel_scale"] = float(PIXEL_SCALE)
        f.attrs["numpix"] = int(NUMPIX)
        f.attrs["z_lens_source"] = (float(z_lens), float(z_source))
        f.attrs["mass_range"] = MASS_RANGE
        f.attrs["sub_pos_range"] = SUB_POS_RANGE
        f.attrs["src_pos_range"] = SRC_POS_RANGE

        f.attrs["compute_chi2"] = bool(COMPUTE_CHI2)
        f.attrs["compute_delta"] = bool(COMPUTE_DELTA)
        f.attrs["compute_bic"] = bool(COMPUTE_BIC)

        if COMPUTE_DELTA:
            f.attrs["delta_chi2_definition"] = "dchi2 = chi2_nosub - chi2_sub; dchi2r = chi2r_nosub - chi2r_sub (positive favors sub)"

        if COMPUTE_BIC:
            f.attrs["bic_definition"] = "BIC = chi2 + k ln(n); dBIC = BIC_nosub - BIC_sub"
            f.attrs["k_sub"] = int(K_SUB)
            f.attrs["k_nosub"] = int(K_NOSUB)

    print(f"\nDataset saved as {OUT_NAME}")
