# Rebuild the checked-in MIX_PLUS design with Vivado 2025.2.
# Run from any working directory:
#   vivado -mode batch -source scripts/rebuild_hardware.tcl
# Dependency/IP generation check without synthesis:
#   vivado -mode batch -source scripts/rebuild_hardware.tcl -tclargs --check
# Full build accepts: -tclargs --jobs 4
set repo_root [file normalize [file join [file dirname [info script]] ..]]
set check_only 0
set jobs 4
for {set i 0} {$i < [llength $argv]} {incr i} {
    set arg [lindex $argv $i]
    switch -- $arg {
        --check {set check_only 1}
        --jobs {
            incr i
            set jobs [lindex $argv $i]
            if {![string is integer -strict $jobs] || $jobs < 1} {error "--jobs requires a positive integer"}
        }
        default {error "Unknown argument: $arg"}
    }
}
if {![string match "2025.2*" [version -short]]} {
    error "This source snapshot requires Vivado 2025.2; found [version -short]"
}
set project_dir [file join $repo_root vivado_project ZYNQ_7020_PYNQ 6_ov5640_lcd vivado]
set project_file [file join $project_dir ov5640_lcd_tcu_touch.xpr]
set output_dir [file join $repo_root build hardware]
file mkdir $output_dir
open_project $project_file
set_property ip_repo_paths [file join $repo_root vivado_project ZYNQ_7020_PYNQ ip_repo] [current_project]
update_ip_catalog
set bd_file [get_files -quiet */base.bd]
if {[llength $bd_file] != 1} {error "Expected exactly one base.bd, found $bd_file"}
open_bd_design $bd_file
report_ip_status -file [file join $output_dir ip_status.rpt]
validate_bd_design
save_bd_design
generate_target all $bd_file
# Fail early for absent user inputs, including the TCU module source.
foreach f [get_files -all -quiet] {
    set kind [get_property FILE_TYPE $f]
    if {$kind in {Verilog SystemVerilog VHDL XDC "Block Designs"}} {
        if {![file exists $f]} {error "Required project input is missing: $f"}
    }
}
update_compile_order -fileset sources_1
if {$check_only} {
    puts "HARDWARE_DEPENDENCY_CHECK_PASS"
    close_project
    exit
}
reset_run synth_1
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
set status [get_property STATUS [get_runs synth_1]]
if {![string match "*Complete*" $status]} {error "Synthesis failed: $status"}
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
set status [get_property STATUS [get_runs impl_1]]
if {![string match "*Complete*" $status]} {error "Implementation failed: $status"}
open_run impl_1
report_timing_summary -file [file join $output_dir timing_summary.rpt]
report_utilization -file [file join $output_dir utilization.rpt]
set bits [glob -nocomplain -directory [get_property DIRECTORY [get_runs impl_1]] *.bit]
if {[llength $bits] != 1} {error "Expected one implemented bitstream, found $bits"}
set hwh [file join $project_dir ov5640_lcd_tcu_merged_fix3.gen sources_1 bd base hw_handoff base.hwh]
if {![file exists $hwh]} {error "Generated HWH missing: $hwh"}
file copy -force [lindex $bits 0] [file join $output_dir pefsl_mix_plus_touch.bit]
file copy -force $hwh [file join $output_dir pefsl_mix_plus_touch.hwh]
puts "BUILD_SUCCESS: $output_dir"
close_project
exit
