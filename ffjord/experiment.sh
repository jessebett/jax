#!/bin/bash

# eg 
# ./launch_job.sh gpu ffjord nodes_ffjord.py "--reg=r2 --lam=1" deadline

d=$(date "+%F-%H-%M-%S")
partition=$1
j_name=$2 # job name
file=$3
args=$4 # specify reg and lam
q=$5  # TODO: high, deadline (need account), normal
resource=1
ssd=/scratch/ssd001/home/$USER/jax/ffjord
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
# TODO: input and output
echo -n "#!/bin/bash
. /h/jessebett/.envs/jaxjet/jaxjet.env
touch ${j_dir}/\$SLURM_JOB_ID/DELAYPURGE
python $file $args --seed=0 --dirname=${j_dir}
" > $j_dir/scripts/${j_name}.sh
cp $file $j_dir/scripts/
sbatch $j_dir/scripts/${j_name}.slrm --qos $q
# sbatch $j_dir/scripts/${j_name}.slrm --qos $q --account deadline
