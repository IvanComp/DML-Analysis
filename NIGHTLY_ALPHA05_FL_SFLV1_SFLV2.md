# Nightly Leonardo Study

Prepared batch file: `leonardo_nightly_alpha05_fl_sflv1_sflv2.sbatch`

Planned study:

- datasets: `mnist cifar10 stl10`
- models: `cnn resnet vgg16`
- approaches: `FL SFLV1 SFLV2`
- clients: `100`
- clients per round: `10`
- rounds: `200`
- epochs: `1`
- repetitions: `1`
- alpha: `0.5`
- comparison profile: `fair`
- total scheduled runs: `27`

Submit from the project root:

```bash
export PROJECT_DIR=$PWD
sbatch --account=<YOUR_ACCOUNT> leonardo_nightly_alpha05_fl_sflv1_sflv2.sbatch
```

First run if the virtualenv still has to be created:

```bash
export PROJECT_DIR=$PWD
sbatch --account=<YOUR_ACCOUNT> \
  --export=ALL,PROJECT_DIR=$PWD,CREATE_VENV=1,INSTALL_REQUIREMENTS=1 \
  leonardo_nightly_alpha05_fl_sflv1_sflv2.sbatch
```

Why the script asks for `1 GPU` by default:

- this code launches one study process and does not split a single run across multiple GPUs
- in `flower_baseline.py`, Flower/Ray assigns fractional GPU resources to simulated clients, but the total budget per run still fits within one GPU
- requesting `2 GPU` is mainly useful if you later split the workload into multiple independent jobs, not for speeding up this single serial study as-is
