import glori.analysis.ldm.io as ldmio
import glori.analysis.ldm.bdsf as ldmbdsf

# Load LDM Sampling result
run_name = "oscillating-crankshaft"

res = ldmio.load_sampling_result(run_name, missing_is_error=False)

# Extract data from sweep results
sampled_imgs = res["img_batch"]  # Sampled images
original_imgs = res["original_imgs"]  # Original images (unscaled)
srls = res["srls"]  # List of pandas df output source catalogs
wcs = res["wcs"]  # List of WCS objects
pos_masks = res["pos_masks"]  # List of boolean masks for input sources
input_catalogs = res["input_catalogs"]
ctxt = res["ctxt_map"]


# Run pybdsf
big_img_cut_batch = original_imgs.reshape(-1, *original_imgs.shape[-2:])
bdsf_folder = res["res_parent"] / "bdsf"
bdsf_workers = 16

original_results, n_successful_original_images, n_failed_original_images = (
    ldmbdsf.run_bdsf_parallel(
        big_img_cut_batch,
        bdsf_folder,
        logger=None,
        max_workers=bdsf_workers,
        file_prefix="original",
    )
)
