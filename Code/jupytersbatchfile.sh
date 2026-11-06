#!/bin/bash
#SBATCH --partition=dgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=desktop[6-8]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH --output jupyter-notebook-%J.log

XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)

echo -e "
==================================================================
NODE_NAME: ${node}
PORT_NUMBER: ${port}
==================================================================
"

source ~/.venvs/gicu_env/bin/activate

# Start the server listening on all network interfaces
jupyter-notebook --no-browser --port=${port} --ip=0.0.0.0