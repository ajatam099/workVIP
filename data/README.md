# VIP Dataset Collection

This directory contains curated datasets for benchmarking the Vision Inspection Pipeline. Datasets are organized by licensing terms to ensure proper attribution and compliance.

## 📁 Directory Structure

```
data/
├── raw/                          # Downloaded datasets (original format)
│   ├── mvtec_ad/
│   ├── roboflow_plastic_defects/
│   └── [other_datasets]/
├── processed/                    # Harmonized manifests and splits
│   ├── mvtec_ad/
│   │   └── manifest.jsonl
│   └── [other_datasets]/
├── download_registry.jsonl       # Download provenance tracking
└── README.md                     # This file
```

## 🔓 Section A: CC0/Permissive Datasets (Safe for Broad Use)

These datasets have permissive licenses allowing commercial use, modification, and redistribution with minimal restrictions.

| Dataset | License | Defect Types | Size | Description |
|---------|---------|--------------|------|-------------|
| [AFID Fabric Defects](https://www.aitex.es/afid) | **CC0 1.0** (Public Domain) | 12 fabric defects: fuzzy ball, nep, broken end, contamination, etc. | 247 images (4096×256px) | High-resolution fabric defect images with pixel-level masks |
| [NEU-CLS Steel Defects](https://figshare.com/articles/dataset/NEU_surface_defect_database/1035131) | **CC BY 4.0** | 6 steel defects: crazing, scratch, pitted surface, patches, rolled-in scale, inclusion | 1,800 images (200×200px) | Classic steel surface defect classification dataset |
| [DAGM 2007](https://zenodo.org/record/60814) | **CC BY 4.0** | Synthetic texture defects (elliptical) | 16,100 grayscale images | Industrial optical inspection benchmark with 10 texture classes |
| [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) | **CC BY 4.0** | Concrete surface cracks | 56,000 images (256×256px) | Large-scale crack detection dataset from bridges, walls, pavements |
| [GC10-DET](https://www.kaggle.com/datasets/alex000kim/gc10det) | **CC BY 4.0** | 10 metal defects: punching hole, welding line, water spot, oil spot, etc. | 2,300 images (~919MB) | Metal surface defect detection with bounding box annotations |
| [Roboflow: Plastic Defects](https://universe.roboflow.com/panops/plastic-defects) | **CC BY 4.0** | Black speck, dirty mark, scratches on plastic containers | 149 images | Directly relevant to VIP use case - plastic container defects |
| [Roboflow: Defect Container Detection](https://universe.roboflow.com/hassan-muhammadnizar/defect-container-detection) | **CC BY 4.0** | General container defects (single "Defected" class) | 385 images | Container defect detection with bounding boxes |
| [Roboflow: Glass Defects](https://universe.roboflow.com/yusuf-zkan/glass-defects-obkeo) | **CC BY 4.0** | Glass surface defects: broken, crack, scratch | 174 images | Glass defect detection relevant for transparent containers |

### Usage Notes for Section A:
- ✅ **Commercial use allowed**
- ✅ **Modification and redistribution permitted**  
- ✅ **Attribution appreciated but not required** (except CC BY variants)
- ✅ **Safe for thesis publication and academic use**

---

## ⚠️ Section B: Attribution-Required/Restricted Datasets

These datasets require attribution, have non-commercial restrictions, or other usage limitations. **Read license terms carefully.**

| Dataset | License | Defect Types | Size | Description |
|---------|---------|--------------|------|-------------|
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | **CC BY-NC-SA 4.0** ⚠️ | 73 defect types across 15 categories including bottle & capsule | 5,354 high-res images | **NON-COMMERCIAL ONLY** - Industrial anomaly detection benchmark |
| [MVTec LOCO AD](https://www.mvtec.com/company/research/datasets/mvtec-loco) | **CC BY-NC-SA 4.0** ⚠️ | Logical/structural anomalies: wrong fill level, missing labels, contamination | 3,651 images | **NON-COMMERCIAL ONLY** - Advanced anomaly detection |
| [Kolektor SDD](https://www.vicos.si/resources/kolektorsdd/) | **CC BY-NC-SA 4.0** ⚠️ | Electronic commutator defects: scratches, spots | 399 images (~500×1240px) | **NON-COMMERCIAL ONLY** - Industrial production images |
| [Kolektor SDD2](https://www.vicos.si/resources/kolektorsdd2/) | **CC BY-NC-SA 4.0** ⚠️ | Surface defects: scratches, spots, imperfections | 3,335 images | **NON-COMMERCIAL ONLY** - Extended KSDD dataset |
| [ELPV Solar Cell Dataset](https://github.com/zae-bayern/elpv-dataset) | **CC BY-NC-SA 4.0** ⚠️ | Solar cell defects (functional vs. defective) | 2,624 grayscale images (300×300px) | **NON-COMMERCIAL ONLY** - Photovoltaic defect detection |
| [Roboflow: Scratches-and-Dents](https://universe.roboflow.com/anomlydetection/scratches-and-dents-detection) | **MIT** | Broken glass, dent, dislocation, scratch, shatter | 268 images | MIT license - attribution required |

### Special Cases:
| Dataset | License/Status | Notes |
|---------|----------------|-------|
| [Severstal Steel Defect](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data) | **Competition Rules** | Kaggle competition data - check specific terms |
| [NEU-DET](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) | **Unknown** | License unclear - use with caution |
| [Magnetic Tile Defects](https://github.com/abin24/Magnetic-tile-defect-datasets) | **Unknown** | GitHub repo without explicit license |
| [CPLID Power-Line Insulators](https://github.com/njuvision/CPLID) | **Not Specified** | Academic dataset - contact authors for commercial use |

### Usage Notes for Section B:
- ⚠️ **NON-COMMERCIAL datasets**: Only for research/academic use
- ⚠️ **Attribution required**: Must cite in publications and derivative works
- ⚠️ **Share-alike**: Derivatives must use same license
- ⚠️ **Unknown licenses**: Use with extreme caution, contact authors

---

## 📚 Required Citations

### For MVTec Datasets:
```bibtex
@article{bergmann2019mvtec,
  title={MVTec AD--A comprehensive real-world dataset for unsupervised anomaly detection},
  author={Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9592--9600},
  year={2019}
}

@article{bergmann2021mvtec,
  title={The MVTec LOCO anomaly detection dataset},
  author={Bergmann, Paul and Batzner, Kilian and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  journal={arXiv preprint arXiv:2111.13677},
  year={2021}
}
```

### For Kolektor Datasets:
```bibtex
@article{tabernik2020segmentation,
  title={Segmentation-based deep-learning approach for surface-defect detection},
  author={Tabernik, Domen and {\v{S}}ela, Samo and Skvar{\v{c}}, Jure and Sko{\v{c}}aj, Danijel},
  journal={Journal of Intelligent Manufacturing},
  volume={31},
  number={3},
  pages={759--776},
  year={2020}
}
```

### For SDNET2018:
```bibtex
@article{maguire2018sdnet2018,
  title={SDNET2018: A concrete crack image dataset for machine learning applications},
  author={Maguire, Marc and Dorafshan, Sattar and Thomas, Robert J},
  year={2018},
  publisher={Utah State University}
}
```

---

## 🚀 Quick Start

### 1. Download Datasets
```bash
# List available datasets
python scripts/download_datasets.py --list

# Download all permissive datasets
python scripts/download_datasets.py --dry-run  # Preview first
python scripts/download_datasets.py

# Download specific dataset
python scripts/download_datasets.py --only mvtec_ad
```

### 2. Install Dependencies
```bash
# For Kaggle datasets
pip install kaggle
kaggle config  # Set up API credentials

# For direct downloads
pip install requests
```

### 3. Manual Downloads
Some datasets require manual steps:
- **Roboflow datasets**: Create free account, download in YOLO/COCO format
- **MVTec datasets**: Direct download from official website
- **Kaggle competitions**: Join competition first, then download

### 4. Create Manifests
```bash
python scripts/create_manifests.py  # Creates processed/*/manifest.jsonl
```

### 5. Run Benchmarks
```bash
python scripts/bench_run.py --config bench/configs/experiments/multi_dataset_comparison.yaml
```

---

## 📋 Dataset Registry

The `download_registry.jsonl` file tracks all downloaded datasets with metadata:

```json
{
  "dataset": "mvtec_ad",
  "version": null,
  "source_url": "https://www.mvtec.com/company/research/datasets/mvtec-ad",
  "license": "CC BY-NC-SA 4.0",
  "downloaded_at": "2025-09-20T15:30:00.000000",
  "sha256": "computed_hash",
  "status": "success",
  "size_mb": 4800,
  "defect_types": ["scratches", "dents", "contamination", "cracks"]
}
```

---

## 🔍 Dataset Quality Notes

### High Quality (Recommended):
- **MVTec AD/LOCO**: Industry standard, high-resolution, pixel-precise annotations
- **SDNET2018**: Large scale, diverse crack examples
- **GC10-DET**: Professional annotations, multiple defect types
- **Roboflow Plastic Defects**: Directly relevant to VIP use case

### Medium Quality:
- **NEU datasets**: Smaller scale but well-established in literature
- **Magnetic Tile**: Good for segmentation but limited domain
- **DAGM 2007**: Synthetic but useful for algorithm validation

### Use with Caution:
- **Unknown license datasets**: Legal risks for commercial use
- **Competition datasets**: May have usage restrictions
- **GitHub repos without licenses**: Copyright unclear

---

## 📞 Support & Contributions

- **Issues**: Report broken links or licensing questions via GitHub issues
- **New datasets**: Submit PRs with dataset info following the table format
- **License questions**: Always verify license terms independently

**Remember**: When in doubt about licensing, contact dataset authors directly or consult legal counsel for commercial applications.
