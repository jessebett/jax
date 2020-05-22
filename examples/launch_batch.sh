#!/bin/bash
dir=$(date "+%F-%H-%M-%S")

partition=$1
j_name=$2
file=$3
# args=$4 too hard to specify no args
q=$4

# lam sweeps
# TODO: reg name!
reg=r2
r_lam_file=${reg}_lams.txt

mkdir -p vaughan2/$dir/$j_name
cp $r_lam_file vaughan2/$dir/$j_name  # TODO: check this, and also this is hardcoded!

while IFS= read -r lam
do
    ./launch_job.sh $dir $partition ${j_name}_${reg}_${lam} $file "--lam=$lam --reg=$reg" $q
done < "$r_lam_file"

# plot sweep
# TODO: make sure launch_job is correct as well!
# regs=(none)
# dirnames=(2020-05-08-21-22-39)
# solvers=(heun fehlberg bosh owrenzen)
# for i in "${!regs[@]}"; do
#   reg=${regs[i]}
#   dirname=$(pwd)/${dirnames[i]}
#   for solver in ${solvers[@]}; do
#     ./launch_job.sh $dir $partition ${j_name}_${reg}_${lam}_${solver} $file "--lam=0 --reg=$reg --dirname=$dirname --method=$solver" $q
#     # while IFS= read -r lam
#     # do
#     #   ./launch_job.sh $dir $partition ${j_name}_${reg}_${lam}_${solver} $file "--lam=$lam --reg=$reg --dirname=$dirname --method=$solver" $q
#     # done < "${reg}_lams.txt"
#   done
# done

# plot reg sweep
# TODO
# regs=(none)
# dirnames=(2020-05-08-21-22-39)
# regs_result=(r2 r3 r4 r5)
# for i in "${!regs[@]}"; do
#   reg=${regs[i]}
#   dirname=$(pwd)/${dirnames[i]}
#   for reg_result in ${regs_result[@]}; do
#     lam=0
#     ./launch_job.sh $dir $partition ${j_name}_${reg}_${lam}_${reg_result} $file "--lam=$lam --reg=$reg --dirname=$dirname --reg_result=$reg_result" $q
#     # while IFS= read -r lam
#     # do
#     #   ./launch_job.sh $dir $partition ${j_name}_${reg}_${lam}_${reg_result} $file "--lam=$lam --reg=$reg --dirname=$dirname --reg_result=$reg_result" $q
#     # done < "${reg}_lams.txt"
#   done
# done

# TODO: train vanilla!
# ./launch_job.sh $dir $partition $j_name $file "" $q

