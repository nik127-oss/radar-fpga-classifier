# build_vivado.tcl
# Run in Vivado: vivado -mode batch -source build_vivado.tcl
# OR: Open Vivado GUI -> Tools -> Run Tcl Script

set project_name "pynq_cnn"
set part "xc7z020clg400-1"
set hls_ip_path "./cnn_accel_prj/solution1/impl/ip"

# Create project
create_project $project_name ./$project_name -part $part -force

# Add HLS IP repository
set_property ip_repo_paths $hls_ip_path [current_project]
update_ip_catalog

# Create block design
create_bd_design "design_1"

# Add Zynq PS
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0

# Apply PYNQ-Z2 preset (or configure manually)
# If preset not available, configure manually:
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
] [get_bd_cells processing_system7_0]

# Add CNN accelerator IP
create_bd_cell -type ip -vlnv pynq:hls:cnn_accelerator:1.0 cnn_accelerator_0

# Run connection automation
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" } \
    [get_bd_cells processing_system7_0]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config { Clk_master {/processing_system7_0/FCLK_CLK0} \
              Clk_slave {Auto} \
              Clk_xbar {Auto} \
              Master {/processing_system7_0/M_AXI_GP0} \
              Slave {/cnn_accelerator_0/s_axi_ctrl} \
              intc_ip {New AXI Interconnect} \
              master_apm {0}} \
    [get_bd_intf_pins cnn_accelerator_0/s_axi_ctrl]

# Validate and save
validate_bd_design
save_bd_design

# Create wrapper
make_wrapper -files [get_files design_1.bd] -top
add_files -norecurse ./$project_name/$project_name.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v

# Synthesize + Implement + Generate bitstream
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# Copy outputs for PYNQ
file copy -force ./$project_name/$project_name.runs/impl_1/design_1_wrapper.bit ./cnn_overlay.bit
file copy -force ./$project_name/$project_name.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh ./cnn_overlay.hwh

puts "=== BUILD COMPLETE ==="
puts "Files ready for PYNQ:"
puts "  cnn_overlay.bit"
puts "  cnn_overlay.hwh"
