set tcldir [file dirname [info script]]
source [file join $tcldir project.tcl]

add_files ${project_name}_prj/solution/syn/vhdl
synth_design -top ${project_name} -part $part
report_utilization -file vivado_synth.rpt
