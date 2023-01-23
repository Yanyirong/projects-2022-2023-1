//////////////////////////////////////////////////////////////////////////////////
// Company:        SJTU
// Engineer:       Jinming Zhang
// Create Date:    10:30 06/26/2021 
// Design Name:    
// Module Name:    mul 
// Project Name:   SoC Project
// Target Devices: VC709
// Tool versions:  vivado 2018.3
// Description: 
//
// Dependencies:   
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//////////////////////////////////////////////////////////////////////////////////

module mul#(
    parameter input_width   = 8,
    parameter weight_width  = 8,
    parameter mul_out_width  = 16
    )( 
    input wire  clk, // clock signal
    input wire  rst_n, // reset signal 
    input wire  [input_width   - 1 : 0] data1, 
    input wire  [weight_width  - 1 : 0] data2, 
    output wire [mul_out_width - 1 : 0] result
    );

	wire signed [input_width   - 1 : 0] data1_wire;
	wire signed [weight_width  - 1 : 0] data2_wire;
	reg  signed [mul_out_width - 1 : 0] result_reg;

	assign data1_wire = data1;
	assign data2_wire = data2;

    always@(posedge clk or negedge rst_n)
	begin
		if (!rst_n)
			result_reg <= 0;
        else
            result_reg <= data1_wire * data2_wire;
    end
    
	assign result = result_reg;

endmodule