## PYNQ-Z2 LED Constraints
## LD0-LD3 active-high LEDs

set_property PACKAGE_PIN R14 [get_ports {leds_4bits_tri_o[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {leds_4bits_tri_o[0]}]

set_property PACKAGE_PIN P14 [get_ports {leds_4bits_tri_o[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {leds_4bits_tri_o[1]}]

set_property PACKAGE_PIN N16 [get_ports {leds_4bits_tri_o[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {leds_4bits_tri_o[2]}]

set_property PACKAGE_PIN M14 [get_ports {leds_4bits_tri_o[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {leds_4bits_tri_o[3]}]
