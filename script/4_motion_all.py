import subprocess
from pathlib import Path

# Set paths
lp2p_home = Path.cwd()
lp2p_src = lp2p_home / "src"
sim_source = lp2p_home / "source_videos"
output_dir = lp2p_home / "output"
sim_videos_dir = output_dir / "sim_videos"
DIR_ORG_1P = output_dir / "sim_1P_videos"
DIR_ORG_1P_CP = output_dir / "sim_1P_cropped"

# Set path to filter script
max_min_script = lp2p_src / "max_min_single_frame.py"

# Find all matching TIFF files in the current directory
# Or assign your directory here
tiff_files = DIR_ORG_1P_CP.glob("cc1_t0400.*.tiff")

# Run the filter script for each file
for tiff in tiff_files:
    print(f"Processing: {tiff}")
    subprocess.run(
        ["python", str(max_min_script), "-i", str(tiff)],
        check=True
    )
