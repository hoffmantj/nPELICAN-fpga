# nPELICAN-fpga
c++ HLS code for implementing nanoPELICAN on FPGAs

model_loader.py can be used to extract model weights from a PELICAN-nano .pt model file.
Configuration values can be set at the top build_prj.tcl
The code can be run using 'vitis_hls -f build_prj.tcl'
clean.sh removes files from previous builds.
