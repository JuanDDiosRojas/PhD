# -*- coding: utf-8 -*-
"""
Chunked lensing dataset generator (Lenstronomy -> HDF5).
Optimized for HPC + PyTorch training.

Key features:
- Splits output into multiple HDF5 chunk files (parallelizable).
- Explicit HDF5 chunking for fast minibatch reads.
- Stores images (float32) + physics + metrics (float64 for scalars).
- Primary target: dchi2 (Δχ² = chi2_nosub - chi2_sub).
"""

import os
import argparse
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
# USER CONFIG (defaults)
# ===========================================================
INSTRUMENT   = "EUCLID"    # "LSST" or "EUCLID"
LSST_BAND    = "g"
EUCLID_BAND  = "VIS"
COADD_LSST_YEARS   = 10
COADD_EUCLID_YEARS = 6

# Targets / metrics
COMPUTE_CHI2  = True
COMPUTE_DELTA = True
COMPUTE_BIC   = False

# Images to save (keep what you need for training)
SAVE_IMAGE_OBS         = True
SAVE_SUB_NOISELESS     = True
SAVE_SUB_NOISY         = True
SAVE_NOSUB_NOISELESS   = True
SAVE_NOSUB_NOISY       = True
SAVE_DELTA_PSI         = True
SAVE_SOURCE_ONLY       = True

# ===========================================================
# COSMOLOGY (fixed)
# ===========================================================
z_lens, z_source = 0.881, 2.059
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048)
D_d  = cosmo.angular_diameter_distance(z_lens)
D_s  = cosmo.angular_diameter_distance(z_source)
D_ds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)


# ===========================================================
# INSTRUMENT CONFIG
# ===========================================================
def make_single_band_config(instrument: str):
    inst_name = instrument.upper()
    if inst_name == "LSST":
        inst = LSST(band=LSST_BAND, psf_type="GAUSSIAN", coadd_years=COADD_LSST_YEARS)
        band_label = f"LSST-{LSST_BAND}"
    elif inst_name == "EUCLID":
        inst = Euclid(band=EUCLID_BAND, coadd_years=COADD_EUCLID_YEARS)
        band_label = f"EUCLID-{EUCLID_BAND}"
    else:
        raise ValueError(f"INSTRUMENT must be 'LSST' or 'EUCLID', got '{instrument}'.")
    return inst.kwargs_single_band(), band_label


# ===========================================================
# LENS MODELS (base, fixed)
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

# for BIC if re-enabled
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


def compute_delta_psi(kwargs_lens_sub, kwargs_lens_nosub, stamp_size_arcsec, numpix):
    """
    Delta potential map: psi_sub - psi_nosub on same grid.

    Note: Here we use SHEAR_REDUCED to match the simulation config (optional).
    If you want plain SHEAR instead, change lists below.
    """
    lm_sub   = LensModel(lens_model_list=["EPL", "SIS", "SHEAR_REDUCED"])
    lm_nosub = LensModel(lens_model_list=["EPL", "SHEAR_REDUCED"])
    grid = np.linspace(-stamp_size_arcsec/2, stamp_size_arcsec/2, numpix)
    x, y = np.meshgrid(grid, grid)
    psi_sub   = lm_sub.potential(x, y, kwargs_lens_sub)
    psi_nosub = lm_nosub.potential(x, y, kwargs_lens_nosub)
    return (psi_sub - psi_nosub).astype("f4")


def chi2_vs_model(image_obs, model_nonoise, sim):
    """
    chi^2 = sum ( (obs - model)^2 / sigma^2 )
    sigma estimated from observed image via sim.estimate_noise.
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
def simulate_single_band(m_sub, pos_sub, pos_src, *,
                         numpix, kwargs_band, kwargs_numerics,
                         stamp_size_arcsec):
    # Build simulators (same band, different lens model lists)
    sim_sub   = SimAPI(numpix=numpix, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model_sub)
    sim_nosub = SimAPI(numpix=numpix, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model_nosub)

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

    # noisy versions
    model_sub_noisy   = model_sub_nonoise   + sim_sub.noise_for_model(model_sub_nonoise)
    model_nosub_noisy = model_nosub_nonoise + sim_nosub.noise_for_model(model_nosub_nonoise)

    # observed image: assume sky is SUB world
    image_obs = model_sub_nonoise + sim_sub.noise_for_model(model_sub_nonoise)

    # delta-psi
    delta_psi = compute_delta_psi(kwargs_lens_sub, kwargs_lens_nosub,
                                  stamp_size_arcsec=stamp_size_arcsec, numpix=numpix) if SAVE_DELTA_PSI else None

    # source-only
    if SAVE_SOURCE_ONLY:
        kwargs_model_source = {"lens_model_list": [], "source_light_model_list": ["SERSIC_ELLIPSE"]}
        sim_source = SimAPI(numpix=numpix, kwargs_single_band=kwargs_band, kwargs_model=kwargs_model_source)
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
        "image_sub_nonoise":    model_sub_nonoise.astype("f4"),
        "image_sub_noisy":      model_sub_noisy.astype("f4"),
        "image_nosub_nonoise":  model_nosub_nonoise.astype("f4"),
        "image_nosub_noisy":    model_nosub_noisy.astype("f4"),
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
# HDF5 WRITER
# ===========================================================
def create_datasets(f: h5py.File, n: int, numpix: int, *, img_chunk: int,
                    compression: str, compression_opts: int):
    """
    Create datasets with explicit HDF5 chunking.
    img_chunk corresponds to chunk size along sample axis.
    """
    ds = {}

    # ---- Images ----
    img_chunks = (min(img_chunk, n), numpix, numpix)

    def mk_img(name):
        return f.create_dataset(
            name, (n, numpix, numpix),
            dtype="f4",
            chunks=img_chunks,
            compression=compression,
            compression_opts=compression_opts
        )

    if SAVE_IMAGE_OBS:       ds["image_obs"]           = mk_img("image_obs")
    if SAVE_SUB_NOISELESS:   ds["image_sub_nonoise"]   = mk_img("image_sub_nonoise")
    if SAVE_SUB_NOISY:       ds["image_sub_noisy"]     = mk_img("image_sub_noisy")
    if SAVE_NOSUB_NOISELESS: ds["image_nosub_nonoise"] = mk_img("image_nosub_nonoise")
    if SAVE_NOSUB_NOISY:     ds["image_nosub_noisy"]   = mk_img("image_nosub_noisy")
    if SAVE_DELTA_PSI:       ds["delta_psi"]           = mk_img("delta_psi")
    if SAVE_SOURCE_ONLY:
        ds["source_only_nonoise"] = mk_img("source_only_nonoise")
        ds["source_only_noisy"]   = mk_img("source_only_noisy")

    # ---- Scalars ----
    # (store float64 for metrics; you can change to f4 if you want)
    def mk_scalar(name, dtype="f8"):
        return f.create_dataset(
            name, (n,),
            dtype=dtype,
            chunks=(min(4096, n),),
            compression=compression,
            compression_opts=compression_opts
        )

    # Always store these if computing delta/chi2
    if COMPUTE_CHI2 or COMPUTE_DELTA or COMPUTE_BIC:
        ds["chi2_sub"]    = mk_scalar("chi2_sub", "f8")
        ds["chi2r_sub"]   = mk_scalar("chi2r_sub", "f8")
        ds["chi2_nosub"]  = mk_scalar("chi2_nosub", "f8")
        ds["chi2r_nosub"] = mk_scalar("chi2r_nosub", "f8")

    if COMPUTE_DELTA:
        ds["dchi2"]  = mk_scalar("dchi2", "f8")
        ds["dchi2r"] = mk_scalar("dchi2r", "f8")

    if COMPUTE_BIC:
        ds["BIC_sub"]   = mk_scalar("BIC_sub", "f8")
        ds["BIC_nosub"] = mk_scalar("BIC_nosub", "f8")
        ds["dBIC"]      = mk_scalar("dBIC", "f8")

    ds["n_pix"] = mk_scalar("n_pix", "i8")

    # ---- Physics ----
    ds["subhalo_mass"] = mk_scalar("subhalo_mass", "f8")
    ds["subhalo_x"]    = mk_scalar("subhalo_x", "f4")
    ds["subhalo_y"]    = mk_scalar("subhalo_y", "f4")
    ds["source_x"]     = mk_scalar("source_x", "f4")
    ds["source_y"]     = mk_scalar("source_y", "f4")

    return ds


def write_attrs(f: h5py.File, *, instrument: str, band_label: str,
                num_samples: int, stamp_size_arcsec: float, pixel_scale: float,
                numpix: int, mass_range, sub_pos_range, src_pos_range,
                seed: int, chunk_id: int, num_chunks: int):
    f.attrs["description"] = f"Single-band lensing dataset ({band_label}) chunked for HPC training."
    f.attrs["instrument"] = instrument.upper()
    f.attrs["band"] = band_label
    f.attrs["num_samples"] = int(num_samples)
    f.attrs["stamp_size_arcsec"] = float(stamp_size_arcsec)
    f.attrs["pixel_scale"] = float(pixel_scale)
    f.attrs["numpix"] = int(numpix)
    f.attrs["z_lens_source"] = (float(z_lens), float(z_source))
    f.attrs["mass_range"] = mass_range
    f.attrs["sub_pos_range"] = sub_pos_range
    f.attrs["src_pos_range"] = src_pos_range
    f.attrs["seed"] = int(seed)
    f.attrs["chunk_id"] = int(chunk_id)
    f.attrs["num_chunks"] = int(num_chunks)

    f.attrs["compute_chi2"] = bool(COMPUTE_CHI2)
    f.attrs["compute_delta"] = bool(COMPUTE_DELTA)
    f.attrs["compute_bic"] = bool(COMPUTE_BIC)

    if COMPUTE_DELTA:
        f.attrs["delta_chi2_definition"] = "dchi2 = chi2_nosub - chi2_sub; dchi2r = chi2r_nosub - chi2r_sub (positive favors sub)"

    if COMPUTE_BIC:
        f.attrs["bic_definition"] = "BIC = chi2 + k ln(n); dBIC = BIC_nosub - BIC_sub"
        f.attrs["k_sub"] = int(K_SUB)
        f.attrs["k_nosub"] = int(K_NOSUB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Output directory for HDF5 chunks.")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Filename prefix (default: auto from instrument/band/settings).")
    parser.add_argument("--n-total", type=int, default=50000,
                        help="Total number of samples across ALL chunks.")
    parser.add_argument("--chunk-size", type=int, default=2500,
                        help="Samples per chunk file. (Ignored if --num-chunks is given.)")
    parser.add_argument("--chunk-id", type=int, default=0,
                        help="Which chunk index to generate (0-based).")
    parser.add_argument("--num-chunks", type=int, default=None,
                        help="Total number of chunks. If set, chunk-size is derived from n-total/num-chunks.")

    parser.add_argument("--seed", type=int, default=1234,
                        help="Base RNG seed. Each chunk uses seed+chunk_id.")
    parser.add_argument("--instrument", type=str, default=INSTRUMENT,
                        choices=["LSST", "EUCLID"], help="Instrument config.")
    parser.add_argument("--stamp-arcsec", type=float, default=6.464,
                        help="Stamp size in arcsec (used to set NUMPIX).")

    parser.add_argument("--mass-min", type=float, default=1e6)
    parser.add_argument("--mass-max", type=float, default=1e9)
    parser.add_argument("--subpos-min", type=float, default=-1.6)
    parser.add_argument("--subpos-max", type=float, default=1.6)
    parser.add_argument("--srcpos-min", type=float, default=-1.0)
    parser.add_argument("--srcpos-max", type=float, default=1.0)

    parser.add_argument("--compression", type=str, default="gzip",
                        choices=["gzip", "lzf"], help="HDF5 compression.")
    parser.add_argument("--compression-opts", type=int, default=4,
                        help="Compression level for gzip (ignored for lzf).")
    parser.add_argument("--img-chunk", type=int, default=64,
                        help="HDF5 chunk size along sample axis for image datasets (tune to batch size).")

    args = parser.parse_args()

    kwargs_band, band_label = make_single_band_config(args.instrument)
    pixel_scale = float(kwargs_band["pixel_scale"])

    stamp_size_arcsec = float(args.stamp_arcsec)
    numpix = int(round(stamp_size_arcsec / pixel_scale))
    kwargs_numerics = {"point_source_supersampling_factor": 1}

    # Ranges
    MASS_RANGE    = (float(args.mass_min), float(args.mass_max))
    SUB_POS_RANGE = (float(args.subpos_min), float(args.subpos_max))
    SRC_POS_RANGE = (float(args.srcpos_min), float(args.srcpos_max))

    # Determine chunking
    if args.num_chunks is not None:
        num_chunks = int(args.num_chunks)
        # distribute as evenly as possible
        base = args.n_total // num_chunks
        rem  = args.n_total % num_chunks
        # first 'rem' chunks have one extra
        n_this = base + (1 if args.chunk_id < rem else 0)
        start = args.chunk_id * base + min(args.chunk_id, rem)
    else:
        chunk_size = int(args.chunk_size)
        num_chunks = int(np.ceil(args.n_total / chunk_size))
        start = args.chunk_id * chunk_size
        n_this = max(0, min(chunk_size, args.n_total - start))

    if n_this <= 0:
        raise ValueError(f"Chunk {args.chunk_id} has zero samples (start={start}). Check chunk settings.")

    # Filename
    if args.prefix is None:
        tag_parts = [args.instrument.upper(), band_label, "singleband", "dCHI2"]
        prefix = "_".join(tag_parts)
    else:
        prefix = args.prefix

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = f"{prefix}_chunk_{args.chunk_id:04d}.h5"
    out_path = os.path.join(args.out_dir, out_name)

    # RNG per chunk (reproducible)
    rng = np.random.default_rng(args.seed + args.chunk_id)

    # Write file
    with h5py.File(out_path, "w") as f:
        ds = create_datasets(
            f, n_this, numpix,
            img_chunk=args.img_chunk,
            compression=args.compression,
            compression_opts=args.compression_opts
        )

        # Generate and write
        for i in range(n_this):
            m_sub = 10 ** rng.uniform(np.log10(MASS_RANGE[0]), np.log10(MASS_RANGE[1]))
            x_sub = rng.uniform(*SUB_POS_RANGE)
            y_sub = rng.uniform(*SUB_POS_RANGE)
            x_src = rng.uniform(*SRC_POS_RANGE)
            y_src = rng.uniform(*SRC_POS_RANGE)

            out = simulate_single_band(
                m_sub, (x_sub, y_sub), (x_src, y_src),
                numpix=numpix, kwargs_band=kwargs_band,
                kwargs_numerics=kwargs_numerics,
                stamp_size_arcsec=stamp_size_arcsec
            )

            # images
            if "image_obs" in ds:           ds["image_obs"][i]           = out["image_obs"]
            if "image_sub_nonoise" in ds:   ds["image_sub_nonoise"][i]   = out["image_sub_nonoise"]
            if "image_sub_noisy" in ds:     ds["image_sub_noisy"][i]     = out["image_sub_noisy"]
            if "image_nosub_nonoise" in ds: ds["image_nosub_nonoise"][i] = out["image_nosub_nonoise"]
            if "image_nosub_noisy" in ds:   ds["image_nosub_noisy"][i]   = out["image_nosub_noisy"]
            if "delta_psi" in ds and out["delta_psi"] is not None:
                ds["delta_psi"][i] = out["delta_psi"]
            if "source_only_nonoise" in ds and out["source_only_nonoise"] is not None:
                ds["source_only_nonoise"][i] = out["source_only_nonoise"]
            if "source_only_noisy" in ds and out["source_only_noisy"] is not None:
                ds["source_only_noisy"][i] = out["source_only_noisy"]

            # scalars / metrics
            if "chi2_sub" in ds:
                ds["chi2_sub"][i]    = out["chi2_sub"]
                ds["chi2r_sub"][i]   = out["chi2r_sub"]
                ds["chi2_nosub"][i]  = out["chi2_nosub"]
                ds["chi2r_nosub"][i] = out["chi2r_nosub"]

            if "dchi2" in ds:
                ds["dchi2"][i]  = out["dchi2"]
                ds["dchi2r"][i] = out["dchi2r"]

            if "BIC_sub" in ds:
                ds["BIC_sub"][i]   = out["BIC_sub"]
                ds["BIC_nosub"][i] = out["BIC_nosub"]
                ds["dBIC"][i]      = out["dBIC"]

            ds["n_pix"][i] = out["n_pix"]

            # physics
            ds["subhalo_mass"][i] = m_sub
            ds["subhalo_x"][i]    = x_sub
            ds["subhalo_y"][i]    = y_sub
            ds["source_x"][i]     = x_src
            ds["source_y"][i]     = y_src

            if (i + 1) % 50 == 0:
                print(f"[chunk {args.chunk_id:04d}] {i+1}/{n_this}  dchi2={out['dchi2']:.3f}  dchi2r={out['dchi2r']:.6f}")

        write_attrs(
            f,
            instrument=args.instrument,
            band_label=band_label,
            num_samples=n_this,
            stamp_size_arcsec=stamp_size_arcsec,
            pixel_scale=pixel_scale,
            numpix=numpix,
            mass_range=MASS_RANGE,
            sub_pos_range=SUB_POS_RANGE,
            src_pos_range=SRC_POS_RANGE,
            seed=args.seed,
            chunk_id=args.chunk_id,
            num_chunks=num_chunks
        )

    print(f"\nSaved: {out_path}")
    print(f"Chunk {args.chunk_id}/{num_chunks-1}  n={n_this}  numpix={numpix}  pixel_scale={pixel_scale:.4f}")


if __name__ == "__main__":
    main()
