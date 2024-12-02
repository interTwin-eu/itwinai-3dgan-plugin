# 3DGAN use case

**Integration author(s)**: Kalliopi Tsolaki (CERN), Matteo Bunino (CERN)

First, create a Python virtual environment, then install the plugin.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Installation in editable mode
pip install --no-cache-dir -e .
```

> [!NOTE]
> Python commands below assumed to be executed from within the
> virtual environment.

In python you can import the plugin as:

```python
import itwinai.plugins.tdgan
from itwinai.plugins.tdgan.model import Generator
```

## Training

Before you can start training, you have to download the data using the dataloading script:

```bash
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps dataloading_step
```

Now you can launch training using `itwinai` and the provided training configuration `config.yaml`:

```bash
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline
```

The command above shows how to run the training using a single worker,
but if you want to run distributed ML training you have two options: interactive
(launch from terminal) or batch (launch form SLURM job script).

> [!WARNING]
> Before running distributed ML, make sure that the distributed strategy used
> by pytorch lightning is set to `ddp_find_unused_parameters_true` . You can set
> this manually by setting
> `distributed_strategy: ddp_find_unused_parameters_true` in `config.yaml`.

To know more on SLURM, see our [SLURM cheatsheet](https://itwinai.readthedocs.io/latest/getting-started/slurm.html).

### Distributed training on a single node (interactive)

If you want to use SLURM in interactive mode, do the following:

```bash
# Allocate resources (on JSC)
$ salloc --partition=batch --nodes=1 --account=intertwin  --gres=gpu:4 --time=1:59:00
job ID is XXXX
# Get a shell in the compute node (if using SLURM)
$ srun --jobid XXXX --overlap --pty /bin/bash 
# Now you are inside the compute node

# On JSC, you may need to load some modules
ml --force purge
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

# ...before activating the Python environment (adapt this to your env name/path)
source ../../envAI_hdfml/bin/activate
```

To launch the training with torch DDP use:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
    $(which itwinai) exec-pipeline --config config.yaml --pipe-key training_pipeline

# Alternatively, from a SLURM login node:
srun --jobid XXXX --ntasks-per-node=1 torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
    $(which itwinai) exec-pipeline --config config.yaml --pipe-key training_pipeline
```

### Distributed training with SLURM (batch mode)

Differently from the interactive approach, this way allows you to use more than one
compute node, thus allowing to scale the distributed ML to larger resources.

Remember that on JSC there is no internet connection on compute nodes, thus if
your script tries to contact the internet it will fail. If needed, make sure to download
the datasets from the SLURM login node before launching the job.

```bash
# Launch a SLURM batch job (on JSC)
sbatch slurm.jsc.sh

# Launch a SLURM batch job (on Vega)
sbatch slurm.vega.sh

# Check the job in the SLURM queue
squeue -u YOUR_USERNAME

# Check the job status
sacct -j JOBID
```

Job's **stdout** is usually saved to `job.out` and its **stderr** is saved to `job.err`.

### Visualize the results of training

Depending on the logging service that you are using, there are different ways to inspect
the logs generated during ML training.

To visualize the logs generated with **MLFLow**, if you set a local path as tracking URI,
run the following in the terminal:

```bash
mlflow ui --backend-store-uri LOCAL_TRACKING_URI
```

And select the "3DGAN" experiment.

## Inference

1. As inference dataset we can reuse training/validation dataset,
for instance the one downloaded from Google Drive folder: if the
dataset root folder is not present, the dataset will be downloaded.
The inference dataset is a set of H5 files stored inside `exp_data`
sub-folders:

    ```text
    ├── exp_data
    │   ├── data
    |   │   ├── file_0.h5
    |   │   ├── file_1.h5
    ...
    |   │   ├── file_N.h5
    ```

2. As model, if a pre-trained checkpoint is not available,
we can create a dummy version of it with:

    ```bash
    python create_inference_sample.py
    ```

3. Run inference command. This will generate a `3dgan-generated-data`
folder containing generated particle traces in form of torch tensors
(.pth files) and 3D scatter plots (.jpg images).

    ```bash
    itwinai exec-pipeline --config config.yaml --pipe-key inference_pipeline
    ```

The inference execution will produce a folder called
`3dgan-generated-data` containing
generated 3D particle trajectories (overwritten if already
there). Each generated 3D image is stored both as a
torch tensor (.pth) and 3D scatter plot (.jpg):

```text
├── 3dgan-generated-data
|   ├── energy=1.296749234199524&angle=1.272539496421814.pth
|   ├── energy=1.296749234199524&angle=1.272539496421814.jpg
...
|   ├── energy=1.664689540863037&angle=1.4906378984451294.pth
|   ├── energy=1.664689540863037&angle=1.4906378984451294.jpg
```

However, if `aggregate_predictions` in the `ParticleImagesSaver` step is set to `True`,
only one pickled file will be generated inside `3dgan-generated-data` folder.
Notice that multiple inference calls will create new files under `3dgan-generated-data` folder.

With fields overriding:

```bash
# Override variables
export CERN_DATA_ROOT="../.."  # data root
export TMP_DATA_ROOT=$CERN_DATA_ROOT
export CERN_CODE_ROOT="." # where code and configuration are stored
export MAX_DATA_SAMPLES=20000 # max dataset size
export BATCH_SIZE=1024 # increase to fill up GPU memory
export NUM_WORKERS_DL=4 # num worker processes used by the dataloader to pre-fetch data
export AGGREGATE_PREDS="true" # write predictions in a single file
export ACCELERATOR="gpu" # choose "cpu" or "gpu"
export STRATEGY="auto" # distributed strategy
export DEVICES="0," # GPU devices list


itwinai exec-pipeline --print-config --config $CERN_CODE_ROOT/config.yaml \
    --pipe-key inference_pipeline \
    -o dataset_location=$CERN_DATA_ROOT/exp_data \
    -o logs_dir=$TMP_DATA_ROOT/ml_logs/mlflow_logs \
    -o distributed_strategy=$STRATEGY \
    -o devices=$DEVICES \
    -o hw_accelerators=$ACCELERATOR \
    -o checkpoints_path=$TMP_DATA_ROOT/checkpoints \
    -o inference_model_uri=$CERN_CODE_ROOT/3dgan-inference.pth \
    -o max_dataset_size=$MAX_DATA_SAMPLES \
    -o batch_size=$BATCH_SIZE \
    -o num_workers_dataloader=$NUM_WORKERS_DL \
    -o inference_results_location=$TMP_DATA_ROOT/3dgan-generated-data \
    -o aggregate_predictions=$AGGREGATE_PREDS
```

## Docker image

> [!WARNING]
> This section regarding container images is not up to date.
> For instance, Docker image names are still referring the itwinai library, but
> should reference this repo.

Build from project root with

```bash
# Local
docker buildx build -t itwinai:0.0.1-3dgan-0.1 .

# Ghcr.io
docker buildx build -t ghcr.io/intertwin-eu/itwinai:0.0.1-3dgan-0.1 .
docker push ghcr.io/intertwin-eu/itwinai:0.0.1-3dgan-0.1
```

You can run inference from wherever a sample of H5 files is available
(folder called `exp_data/`'):

```text
├── $PWD    
|   ├── exp_data
|   │   ├── data
|   |   │   ├── file_0.h5
|   |   │   ├── file_1.h5
...
|   |   │   ├── file_N.h5
```

```bash
docker run -it --rm --name running-inference -v "$PWD":/tmp/data ghcr.io/intertwin-eu/itwinai:0.0.1-3dgan-0.1
```

This command will store the results in a folder called `3dgan-generated-data`:

```text
├── $PWD
|   ├── 3dgan-generated-data
|   │   ├── energy=1.296749234199524&angle=1.272539496421814.pth
|   │   ├── energy=1.296749234199524&angle=1.272539496421814.jpg
...
|   │   ├── energy=1.664689540863037&angle=1.4906378984451294.pth
|   │   ├── energy=1.664689540863037&angle=1.4906378984451294.jpg
```

To override fields in the configuration file at runtime, you can use the `-o`
flag. Example: `-o path.to.config.element=NEW_VALUE`.

Please find a complete exampled below, showing how to override default configurations
by setting some env variables:

```bash
# Override variables
export CERN_DATA_ROOT="/usr/data" 
export CERN_CODE_ROOT="/usr/src/app"
export MAX_DATA_SAMPLES=10 # max dataset size
export BATCH_SIZE=64 # increase to fill up GPU memory
export NUM_WORKERS_DL=4 # num worker processes used by the dataloader to pre-fetch data
export AGGREGATE_PREDS="true" # write predictions in a single file
export ACCELERATOR="gpu" # choose "cpu" or "gpu"

docker run -it --rm --name running-inference \
-v "$PWD":/usr/data ghcr.io/intertwin-eu/itwinai:0.0.1-3dgan-0.1 \
/bin/bash -c "itwinai exec-pipeline \
    --print-config --config $CERN_CODE_ROOT/config.yaml \
    --pipe-key inference_pipeline \
    -o dataset_location=$CERN_DATA_ROOT/exp_data \
    -o logs_dir=$TMP_DATA_ROOT/ml_logs/mlflow_logs \
    -o distributed_strategy=$STRATEGY \
    -o devices=$DEVICES \
    -o hw_accelerators=$ACCELERATOR \
    -o checkpoints_path=$TMP_DATA_ROOT/checkpoints \
    -o inference_model_uri=$CERN_CODE_ROOT/3dgan-inference.pth \
    -o max_dataset_size=$MAX_DATA_SAMPLES \
    -o batch_size=$BATCH_SIZE \
    -o num_workers_dataloader=$NUM_WORKERS_DL \
    -o inference_results_location=$TMP_DATA_ROOT/3dgan-generated-data \
    -o aggregate_predictions=$AGGREGATE_PREDS "
```

### How to fully exploit GPU resources

Keeping the example above as reference, increase the value of `BATCH_SIZE` as much as possible
(just below "out of memory" errors). Also, make sure that `ACCELERATOR="gpu"`. Also, make sure
to use a dataset large enough by changing the value of `MAX_DATA_SAMPLES` to collect meaningful
performance data. Consider that each H5 file contains roughly 5k items, thus setting
`MAX_DATA_SAMPLES=10000` should be enough to use all items in each input H5 file.

You can try:

```bash
export MAX_DATA_SAMPLES=10000 # max dataset size
export BATCH_SIZE=1024 # increase to fill up GPU memory
export ACCELERATOR="gpu
```

### Singularity

Run Docker container with Singularity:

```bash
singularity run --nv -B "$PWD":/usr/data docker://ghcr.io/intertwin-eu/itwinai:0.0.1-3dgan-0.1 /bin/bash -c \
"cd /usr/src/app && itwinai exec-pipeline --config config.yaml --pipe-key inference_pipeline"
```

Example with overrides (as above for Docker):

```bash
# Override variables
export CERN_DATA_ROOT="/usr/data" 
export CERN_CODE_ROOT="/usr/src/app"
export MAX_DATA_SAMPLES=10 # max dataset size
export BATCH_SIZE=64 # increase to fill up GPU memory
export NUM_WORKERS_DL=4 # num worker processes used by the dataloader to pre-fetch data
export AGGREGATE_PREDS="true" # write predictions in a single file
export ACCELERATOR="gpu" # choose "cpu" or "gpu"

singularity run --nv -B "$PWD":/usr/data docker://ghcr.io/intertwin-eu/itwinai:0.0.1-3dgan-0.1 /bin/bash -c \
"cd /usr/src/app && itwinai exec-pipeline \
    --print-config --config $CERN_CODE_ROOT/config.yaml \
    --pipe-key inference_pipeline \
    -o dataset_location=$CERN_DATA_ROOT/exp_data \
    -o logs_dir=$TMP_DATA_ROOT/ml_logs/mlflow_logs \
    -o distributed_strategy=$STRATEGY \
    -o devices=$DEVICES \
    -o hw_accelerators=$ACCELERATOR \
    -o checkpoints_path=$TMP_DATA_ROOT/checkpoints \
    -o inference_model_uri=$CERN_CODE_ROOT/3dgan-inference.pth \
    -o max_dataset_size=$MAX_DATA_SAMPLES \
    -o batch_size=$BATCH_SIZE \
    -o num_workers_dataloader=$NUM_WORKERS_DL \
    -o inference_results_location=$TMP_DATA_ROOT/3dgan-generated-data \
    -o aggregate_predictions=$AGGREGATE_PREDS "
```
