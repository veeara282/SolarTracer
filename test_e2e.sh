#!/bin/sh
# Run an end-to-end test of experiment.py (not a real experiment)
python experiment.py -n 3 --train-dir SPI_val -o test.pt --logfile test.json -e 3
