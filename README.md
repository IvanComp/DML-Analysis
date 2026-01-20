# FedSSIM: Federated Learning with Grad-CAM Refinement

Questo progetto simula un processo di Federated Learning utilizzando **FedAvg** e un meccanismo di raffinamento avanzato denominato **FedGradCAM**.

Il sistema confronta un'aggregazione standard dei pesi con un'aggregazione pesata basata sulle "regioni di interesse" identificate tramite **Grad-CAM** sui singoli client.

## Requisiti

- Python 3.8 o superiore
- pip

## Configurazione Ambiente

Per creare un ambiente virtuale pulito e installare le dipendenze, esegui i seguenti comandi:

```bash
# 1. Crea l'ambiente virtuale
python3 -m venv venv

# 2. Attiva l'ambiente virtuale
# Su macOS/Linux:
source venv/bin/activate
# Su Windows:
# venv\Scripts\activate

# 3. Aggiorna pip e installa le dipendenze
pip install --upgrade pip
pip install -r requirements.txt
```

## Esecuzione Esperimenti

Lo script principale è `fed_gradcam.py`. Puoi configurare l'esperimento tramite argomenti da riga di comando.

### Comandi principali

Esegui una simulazione standard con STL-10 (128x128) e dati Non-IID:

```bash
python fed_gradcam.py --dataset stl10 --num_clients 4 --distribution non-iid --alpha 0.5
```

Esegui una simulazione con CIFAR-10 e dati IID:

```bash
python fed_gradcam.py --dataset cifar10 --num_clients 4 --distribution iid
```

### Parametri disponibili

- `--dataset`: `stl10` o `cifar10` (default: `stl10`)
- `--num_clients`: Numero di client (default: `4`)
- `--distribution`: `iid` o `non-iid` (default: `non-iid`)
- `--alpha`: Parametro di Dirichlet per la distribuzione Non-IID (default: `0.5`)
- `--rounds`: Numero di round di comunicazione (default: `5`)
- `--epochs`: Epoche locali per ogni client (default: `2`)
- `--lr`: Learning rate (default: `0.01`)

## Output e Risultati

Lo script genera automaticamente i seguenti file dopo ogni esecuzione:

1. **`visualizations/`**: Cartella contenente le heatmap Grad-CAM generate dai client (immagini originali e overlay).
2. **`channel_importance_log.csv`**: Log dei pesi di importanza dei canali utilizzati dal server per l'aggregazione FedGradCAM.
3. **`simulation_comparison_[dataset]_[dist].png`**: Grafici comparativi di F1-Score e tempi di addestramento tra FedAvg e FedGradCAM.

## Struttura del progetto

- `fed_gradcam.py`: Script principale contenente la logica di training, Grad-CAM e simulazione FL.
- `requirements.txt`: Elenco delle librerie Python necessarie.
- `.gitignore`: Configurazione per escludere dati e dataset pesanti dal controllo di versione.
