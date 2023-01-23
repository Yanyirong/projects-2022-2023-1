//////////////////////////////////////////////////////////////////////////////////
// Company:        SJTU
// Engineer:       Jinming Zhang
// Create Date:    10:50 10/16/2021 
// Design Name:    
// Module Name:    line_buffer 
// Project Name:   Huawei Bei
// Target Devices: VC709
// Tool versions:  vivado 2018.3
// Description: 
// Dependencies:  
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//////////////////////////////////////////////////////////////////////////////////

module line_buffer#(
        parameter addr_width    = 7,
        parameter data_width    = 8,
        parameter data_depth    = 128
    ) (
        input  wire                       clka,  // system clk
        input  wire                       ena,   // enable signal, high valid
        input  wire                       wena,   // enable signal, high valid
        input  wire  [addr_width   -1:0]  addra, // addr of ram_inst
        input  wire  [data_width   -1:0]  dina,   // enable signal, high valid
        output reg   [data_width   -1:0]  douta  // ram output for inst
    );

    ///////////////////////////////////////////////////////////////////////////
    // internal signal definition
    ///////////////////////////////////////////////////////////////////////////

    reg  [data_width  - 1 : 0]  ram[0 : data_depth - 1]; // ram output for inst
    reg  [addr_width  - 1 : 0]  addra_reg;

    ///////////////////////////////////////////////////////////////////////////
    // function
    ///////////////////////////////////////////////////////////////////////////
    always @(posedge clka)
    begin
        if(ena & wena)
        begin
            ram[addra] <= dina;
            douta <= ram[addra];
        end
        else 
        begin
            douta <= ram[addra];
        end

    end
  ///////////////////////////////////////////////////////////////////////////
  //
  ///////////////////////////////////////////////////////////////////////////
endmodule

/////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////

