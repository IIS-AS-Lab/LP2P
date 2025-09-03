## 🧪 LP2P Calcium Imaging Simulation Pipeline
This repository provides a simulation pipeline for generating synthetic calcium imaging data by combining 1-photon (1P) background videos with 2-photon (2P) neuron foreground videos.  
The pipeline handles motion simulation, cropping, filtering, and file management in a structured and reproducible way.


## 📁 Project Structure
```
├── run_simulation.py             # Main entry point for the simulation
├── environment.yml               # Conda environment dependencies
├── src/                          # Python scripts for filtering and simulation
├── images/                       # 
│   ├── pipeline_flowchart.png	  # Flowchart image
├── source_videos/                # Input TIFF videos (1P, 2P, background)
├── output/                       # All simulation output goes here
│   ├── sim_1P_videos/            # 1-photon raw simulated videos
│   ├── sim_1P_cropping_videos/   # 1-photon cropped videos
│   └── sim_videos/               # Final processed videos
├── LICENSE                 	  # MIT License details
└── README.md                     # Project documentation
```

## 📋 Prerequisites
- Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
or [anaconda](https://www.anaconda.com/products/distribution).
- Ensure [git](https://git-scm.com/) is installed to clone the repository.
```bash
git clone https://github.com/IIS-AS-Lab/LP2P.git
cd LP2P
```

## ⚙️ Setup
Make sure you have [conda](https://docs.conda.io/) installed.

```bash
conda env create -f environment.yml -n LP2P
conda activate LP2P
```


## 🚀 Usage
python run_video_extract.py
python run_simulation.py --start-motion 1 --end-motion 12

Argument			Description					Default
--start-motion		Starting motion scale		1
--end-motion		Ending motion scale 		12

The generated videos and intermediate results will be saved under the ./output/ folder.


## 🧪 Output Structure
After simulation, results will be organized as follows:
	•	output/sim_1P_videos/ – Simulated videos without cropping or filtering
	•	output/sim_1P_cropping_Videos/ – Cropped simulation frames
	•	output/sim_videos/ – Filtered and finalized videos for analysis or model training


## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

