#!/bin/bash

d=$1
partition=$2
j_name=$3
file=$4
args=$5 # specify reg and lam
q=$6  # TODO: high, deadline (need account), normal
resource=1
ssd=/scratch/ssd001/home/$USER/jessebett/jax/examples  # TODO(jesse): set this to where ffjord.py file is
j_dir=$ssd/vaughan2/$d/$j_name

mkdir -p $j_dir/scripts

# TODO: do we really need 4 cpus?
# build slurm script
mkdir -p $j_dir/log
echo "#!/bin/bash
#SBATCH --job-name=${j_name}
#SBATCH --output=${j_dir}/log/%j.out
#SBATCH --error=${j_dir}/log/%j.err
#SBATCH --partition=${partition}
#SBATCH --cpus-per-task=$[4 * $resource] 
#SBATCH --ntasks-per-node=1
#SBATCH --mem=$[32*$resource]G
#SBATCH --gres=gpu:${resource}
#SBATCH --nodes=1

bash ${j_dir}/scripts/${j_name}.sh
" > $j_dir/scripts/${j_name}.slrm

# build bash script

echo -n "#!/bin/bash
. /h/jkelly/new_jet_nodes.env
touch /checkpoint/$USER/\$SLURM_JOB_ID/DELAYPURGE
python $file $args --seed=0 --dirname=${j_dir} --ckpt_path=/checkpoint/$USER/\$SLURM_JOB_ID/ck.pt

" > $j_dir/scripts/${j_name}.sh

# TODO: this is only for plot version!
# echo -n "#!/bin/bash
# . /h/jkelly/new_jet_nodes.env
# touch /checkpoint/$USER/\$SLURM_JOB_ID/DELAYPURGE
# python $file $args
# 
# " > $j_dir/scripts/${j_name}.sh

# TODO: this is for checking GPU shit
# echo -n "#!/bin/bash
# . /h/jkelly/new_jet_nodes.env
# touch /checkpoint/$USER/\$SLURM_JOB_ID/DELAYPURGE
# nvidia-smi
# " > $j_dir/scripts/${j_name}.sh

cp $file $j_dir/scripts/

# TODO: deadline or nah
# sbatch $j_dir/scripts/${j_name}.slrm --qos $q --account deadline
sbatch $j_dir/scripts/${j_name}.slrm --qos $q
