#!/bin/bash

set -x

LOG="logs/deepseg_`date +%Y-%m-%d_%H-%M-%S`.log"
exec &> >(tee -a "$LOG")

./solve.py
