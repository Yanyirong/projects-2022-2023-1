//////////////////////////////////////////////////////////////////////////////////
// Company:        
// Engineer:       
// Create Date:    
// Design Name:    
// Module Name:    cutoff
// Project Name:   
// Target Devices: 
// Tool versions:  
// Description: 
//
// Dependencies:   
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//////////////////////////////////////////////////////////////////////////////////

module cutoff#(
    parameter input_width        = 20,
    parameter output_width       = 8,
    parameter radix_point_right  = 9
    )( 
    input wire  clk,        // clock signal
    input wire  rst_n,      // reset signal 
    input wire  [input_width  - 1 : 0] data_in, 
    output reg  [output_width - 1 : 0] data_out
    );
    

    wire signed [output_width - 1 : 0] Fix_8_1 ; 
    wire carry ;
    wire [12:0] extend_data;
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
            begin
            end
        else
            begin
                data_out <= Fix_8_1;
            end
    end   
    //assign carry_bit = data_in[19] ? (data_in[7] & (|data_in[6:0]) ): data_in[7] ;
    assign carry = data_in[19] ? (data_in[7] & (data_in[6]|data_in[5]|data_in[4]|data_in[3]|data_in[2]|data_in[1]|data_in[0]) ): data_in[7] ;
    assign extend_data = {data_in[19],data_in[19:8]} + carry ;


    assign Fix_8_1 = (extend_data[12:7] == 6'b000000 || extend_data[12:7] == 6'b111111) ? extend_data[7:0]: 
                {extend_data[12],!extend_data[12],!extend_data[12],!extend_data[12],!extend_data[12],!extend_data[12],!extend_data[12],!extend_data[12]} ;


endmodule