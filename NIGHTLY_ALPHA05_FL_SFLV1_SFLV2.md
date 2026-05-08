# Local And RCM Mini Guide

Questa guida raccoglie i comandi minimi per seguire il progetto ed eseguire gli esperimenti:

- in locale sul proprio computer;
- su Leonardo dentro una sessione RCM interattiva;
- con la configurazione attuale degli esperimenti `FL`, `SFLV1`, `SFLV2` e `alpha=0.5`.

## Configurazione Di Riferimento

- datasets: `mnist cifar10 cifar100`
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

## 1. Setup Locale

Dal root del progetto:

```bash
cd /path/to/DML-Analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verifica rapida dell'ambiente:

```bash
python --version
python flower_baseline.py --help
python run_experiments.py --help
```

## 2. Smoke Test Locale

Test veloce per verificare che la pipeline parta:

```bash
python run_experiments.py \
  --dataset mnist \
  --model cnn \
  --learning-type FL SFLV1 SFLV2 \
  --num-clients 2 \
  --clients-per-round 2 \
  --rounds 2 \
  --epochs 1 \
  --repetitions 1 \
  --data-distr 0.5 \
  --comparison-profile fair \
  --study-name smoke_local_alpha05_fl_sfl
```

## 3. Esperimento Locale Completo

Configurazione locale equivalente a quella preparata per Leonardo:

```bash
python run_experiments.py \
  --dataset mnist cifar10 cifar100 \
  --model cnn resnet vgg16 \
  --learning-type FL SFLV1 SFLV2 \
  --num-clients 100 \
  --clients-per-round 10 \
  --rounds 200 \
  --epochs 1 \
  --repetitions 1 \
  --data-distr 0.5 \
  --comparison-profile fair \
  --study-name nightly_alpha05_fl_sflv1_sflv2_100clients_10cpr_200r_1rep
```

## 4. Dove Vedere I Risultati

Durante o dopo un run:

```bash
ls study_runs
cat study_runs/latest_study.txt
```

Per uno study specifico:

```bash
ls study_runs/<study_name>
ls study_runs/<study_name>/csv
ls study_runs/<study_name>/results
tail -f study_runs/<study_name>/manifest.csv
```

File utili prodotti dal runner:

- `manifest.csv`: stato di ogni run
- `run_summary.csv`: riassunto per run
- `aggregate_metrics.csv`: aggregati
- `pairwise_statistics.csv`: confronti statistici

## 5. Leonardo Da RCM Senza `sbatch`

Dentro RCM non conviene lanciare `python ...` direttamente sul login node. Il flusso corretto è:

1. allocare risorse con `salloc`
2. entrare nel compute node con `srun --pty`
3. lanciare il runner con `srun`

Verifica iniziale:

```bash
hostname
echo $SLURM_JOB_ID
```

Se `SLURM_JOB_ID` è vuoto, alloca una sessione GPU:

```bash
salloc -A INF26_enesma -p boost_usr_prod --qos=normal \
  -N 1 --ntasks=1 --cpus-per-task=32 --gres=gpu:1 \
  -t 24:00:00
```

Quando l'allocazione parte:

```bash
srun --pty bash -l
```

Poi prepara l'ambiente:

```bash
cd /path/to/DML-Analysis

module purge
module load profile/deeplrn
module load $(module -t av cineca-ai 2>&1 | awk '/^cineca-ai\\// {print $1}' | sort -V | tail -n 1)

python -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p slurm
nvidia-smi
```

Comando di esecuzione:

```bash
srun python run_experiments.py \
  --dataset mnist cifar10 cifar100 \
  --model cnn resnet vgg16 \
  --learning-type FL SFLV1 SFLV2 \
  --num-clients 100 \
  --clients-per-round 10 \
  --rounds 200 \
  --epochs 1 \
  --repetitions 1 \
  --data-distr 0.5 \
  --comparison-profile fair \
  --study-name nightly_alpha05_fl_sflv1_sflv2_100clients_10cpr_200r_1rep
```

## 6. Monitoraggio Su Leonardo

Controlli essenziali:

```bash
squeue -u $USER
nvidia-smi
tail -f study_runs/nightly_alpha05_fl_sflv1_sflv2_100clients_10cpr_200r_1rep/manifest.csv
```

Se vuoi controllare i log di un singolo run:

```bash
ls study_runs/nightly_alpha05_fl_sflv1_sflv2_100clients_10cpr_200r_1rep/logs
tail -f study_runs/nightly_alpha05_fl_sflv1_sflv2_100clients_10cpr_200r_1rep/logs/<run_id>.log
```

## 7. Chiusura Sessione

Per uscire dalla shell sul compute node e poi dalla allocazione:

```bash
exit
exit
```

## 8. Nota Sulle GPU

Per questo runner, `1 GPU` è in genere la scelta giusta:

- ogni study viene eseguito come processo seriale;
- il codice non divide un singolo run su più GPU;
- `2 GPU` diventano utili soprattutto se si separano i run in più job indipendenti.

Il file batch preparato in precedenza resta disponibile in [leonardo_nightly_alpha05_fl_sflv1_sflv2.sbatch](/Users/ivan/Desktop/DML-Analysis/leonardo_nightly_alpha05_fl_sflv1_sflv2.sbatch:1), ma questa guida è pensata per il flusso interattivo e locale.
