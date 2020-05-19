#!/bin/bash

for lam in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.5 2.0; do
  bash ./experiment.sh gpu ffjord_r3_lam_${lam} ffjord.py "--reg=r3 --lam=${lam}" normal
done
