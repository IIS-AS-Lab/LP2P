import os
import subprocess
from pathlib import Path

# -------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------

lp2p_home = Path.cwd()
lp2p_src = lp2p_home / "src"
sim_source = lp2p_home / "source_videos"

Neuron_2P_video = "train.04.00.smFG.tiff"
Org_1P_video = "CC1.S10_E3110.tiff"
Org_1P_BK_video = "CC1.S10_E3110.smBK.tiff"

pre_name = "cc1_t0400"
version_prefix = "v0."
start_frame = "10"
end_frame = "2010"
x0, y0, x1, y1 = "132", "122", "567", "577"

motion_scales = [12]
shaking_types = [1, 2, 3]
repetitions = [1, 2, 3]

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------

def run_command(args):
    print(f"Running: {' '.join(map(str, args))}")
    subprocess.run(args, check=True)

# -------------------------------------------------------------------------------------
# Check Files and Generate Background if Needed
# -------------------------------------------------------------------------------------

if (sim_source / Org_1P_video).exists():
    print(f"1P Video File {Org_1P_video} exists.")
    if not (sim_source / Org_1P_BK_video).exists():
        print(f"1P Background Video File {Org_1P_BK_video} does not exist. Generating from {Org_1P_video}...")
        run_command([
            "python3.12", str(lp2p_src/ "E9.3_Tiffs_Butter_Bandpass_Single_RW_Mode.py"),
            "-i", str(sim_source / Org_1P_video),
            "-F_BP_out",
            "-bk_output", Org_1P_BK_video
        ])
else:
    if (sim_source / Org_1P_BK_video).exists():
        print(f"1P Video File {Org_1P_video} not found, but background {Org_1P_BK_video} exists.")
    else:
        raise FileNotFoundError(f"Neither {Org_1P_video} nor {Org_1P_BK_video} exist. Quit!")

if not (sim_source / Neuron_2P_video).exists():
    raise FileNotFoundError(f"2P Neuron File {Neuron_2P_video} does not exist. Quit!")
else:
    print(f"2P Neuron File {Neuron_2P_video} exists.")

# -------------------------------------------------------------------------------------
# Simulation Loop
# -------------------------------------------------------------------------------------

for i in motion_scales:
    for j in shaking_types:
        for k in repetitions:
            snf = f"{version_prefix}{k}"
            pre_file_name = f"{pre_name}.pv{i}.bk{start_frame}_{end_frame}."
            stype = {1: "ns__", 2: "__bk", 3: "nsbk"}[j]
            file_name = f"{pre_file_name}{stype}.{snf}.tiff"
            crop_file_name = f"{pre_file_name}{stype}.{snf}.cp.tiff"
            smFG_crop_file_name = f"{pre_file_name}{stype}.{snf}.cp.smFG.tiff"

            print(f"Generating simulation: {file_name}")

            run_command([
                "python3.12", str(lp2p_src/ "10.10_Simulation_video_generation.py"),
                "-bk_video", str(sim_source / Org_1P_BK_video),
                "-neuron_video", str(sim_source / Neuron_2P_video),
                "-bk_video_max", "2793",
                "-neuron_video_max", "2710",
                "-Up_no", "2",
                "-up_ni_cen_row", "1124",
                "-up_ni_cen_col", "1474",
                "-pv_sigma", str(i),
                "-signal_adjusted_ratio", "0.5",
                "-length", "2000",
                "-prefix_name", pre_name,
                "-suffix_name", snf,
                "-shaking_type", str(j),
                "-start", start_frame,
                "-end", end_frame,
                "-o", file_name,
                "-info", "off"
            ])

            print(f"Cropping: {crop_file_name}")
            run_command([
                "python3.12", str(lp2p_src/ "12.1_Tiff_Block_Crop_SingleFrameRW.py"),
                "-i", file_name,
                "-x0", x0,
                "-y0", y0,
                "-x1", x1,
                "-y1", y1,
                "-noshow",
                "-o", crop_file_name,
                "-save"
            ])

            print(f"Filtering: {smFG_crop_file_name}")
            run_command([
                "python3.12", str(lp2p_src/ "E9.3_Tiffs_Butter_Bandpass_Single_RW_Mode.py"),
                "-i", crop_file_name,
                "-o", smFG_crop_file_name
            ])