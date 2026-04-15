# run_hls.tcl
# Run this in Vivado HLS: vivado_hls -f run_hls.tcl
#
# This script creates the project, runs synthesis, and exports IP.
# If anything hangs, kill the process and try with relaxed pragmas.

open_project cnn_accel_prj
set_top cnn_accelerator

# Add source files
add_files cnn_accel.cpp
add_files cnn_accel.h
add_files weights.h

# Add testbench
add_files -tb cnn_accel_tb.cpp

# Create solution
open_solution "solution1"
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

# Conservative settings to prevent hangs
config_compile -pipeline_style frp
config_schedule -relax_ii 1

# Step 1: C Simulation
puts "=== Running C Simulation ==="
csim_design

# Step 2: C Synthesis
puts "=== Running C Synthesis ==="
csynth_design

# Step 3: Export IP
puts "=== Exporting IP ==="
export_design -format ip_catalog -description "1D CNN Drone Classifier" -vendor "pynq" -library "hls" -version "1.0"

puts "=== DONE ==="
puts "IP exported to: cnn_accel_prj/solution1/impl/ip/"
exit
