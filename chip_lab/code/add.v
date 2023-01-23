//////////////////////////////////////////////////////////////////////////////////
// Company:        SJTU
// Engineer:       Jinming Zhang
// Create Date:    10:30 06/26/2021 
// Design Name:    
// Module Name:    add4, add10 
// Project Name:   SoC Project
// Target Devices: VC709
// Tool versions:  vivado 2018.3
// Description: 
//
// Dependencies:   add4 for 1 pipeline, add10 for 2 pipeline
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//////////////////////////////////////////////////////////////////////////////////


module add2 #(
    parameter input_len     = 16
    ) (
    input   wire                        clk,
    input   wire                        rst_n,
    input   wire    [input_len  -1:0]   data_in1,   // input1
    input   wire    [input_len  -1:0]   data_in2,   // input2
    output  wire    [input_len    :0]   data_out
    );

    ///////////////////////////////////////////////////////////////////////////
    // internal signal definition
    ///////////////////////////////////////////////////////////////////////////

    wire signed     [input_len  -1:0]   data_in1_internal;
    wire signed     [input_len  -1:0]   data_in2_internal;
    reg  signed     [input_len    :0]   data_out_internal;

    ///////////////////////////////////////////////////////////////////////////
    // input
    ///////////////////////////////////////////////////////////////////////////
    assign data_in1_internal = data_in1;
    assign data_in2_internal = data_in2;
    ///////////////////////////////////////////////////////////////////////////
    // function
    ///////////////////////////////////////////////////////////////////////////
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
            data_out_internal <= {1'b0, {input_len{1'b0}}};
        else
            data_out_internal <= data_in1_internal + data_in2_internal;
    end
    // output
    assign data_out = data_out_internal;
endmodule



module add4 #(
    parameter input_len     = 16
    ) (
    input   wire                        clk,
    input   wire                        rst_n,
    input   wire    [input_len  -1:0]   data_in1,   // input1
    input   wire    [input_len  -1:0]   data_in2,   // input2
    input   wire    [input_len  -1:0]   data_in3,   // input3
    input   wire    [input_len  -1:0]   data_in4,   // input4
    output  wire    [input_len  +1:0]   data_out
    );

    ///////////////////////////////////////////////////////////////////////////
    // internal signal definition
    ///////////////////////////////////////////////////////////////////////////

    wire signed     [input_len  -1:0]   data_in1_internal;
    wire signed     [input_len  -1:0]   data_in2_internal;
    wire signed     [input_len  -1:0]   data_in3_internal;
    wire signed     [input_len  -1:0]   data_in4_internal;
    wire signed     [input_len    :0]   data_io1_internal;
    wire signed     [input_len    :0]   data_io2_internal;
    reg  signed     [input_len  +1:0]   data_out_internal;

    ///////////////////////////////////////////////////////////////////////////
    // input
    ///////////////////////////////////////////////////////////////////////////
    assign data_in1_internal = data_in1;
    assign data_in2_internal = data_in2;
    assign data_in3_internal = data_in3;
    assign data_in4_internal = data_in4;
    assign data_io1_internal = data_in1_internal + data_in2_internal;
    assign data_io2_internal = data_in3_internal + data_in4_internal;
    // assign data_out_internal = data_io1_internal + data_io2_internal;

    ///////////////////////////////////////////////////////////////////////////
    // function
    ///////////////////////////////////////////////////////////////////////////
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
            data_out_internal <= {2'b0, {input_len{1'b0}}};
        else
            data_out_internal <= data_io1_internal + data_io2_internal;
    end

    // output
    assign data_out = data_out_internal;
endmodule

module add9 #(
    parameter input_len     = 16
    ) (
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire [input_len * 9 - 1 : 0]     data_in,   // input1
    output wire [input_len + 4 - 1 : 0]     data_out
    );

    ///////////////////////////////////////////////////////////////////////////
    // internal signal definition
    ///////////////////////////////////////////////////////////////////////////

    wire signed     [input_len - 1 : 0]   data_in_internal[9 - 1 : 0];
    wire signed     [input_len + 1 : 0]   data_io0_internal;
    wire signed     [input_len + 1 : 0]   data_io1_internal;

    reg  signed     [input_len - 1 : 0]   data_in_internal8;

    reg  signed     [input_len + 3 : 0]   data_out_internal;
    genvar i;

    ///////////////////////////////////////////////////////////////////////////
    // input
    ///////////////////////////////////////////////////////////////////////////

    generate
    for(i=0; i<9; i=i+1)
    begin
        assign data_in_internal[i] = data_in[input_len * i + input_len - 1: input_len * i];
    end
    endgenerate

    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
            data_in_internal8 <= 0;
        else
            data_in_internal8 <= data_in_internal[8];
    end

    ///////////////////////////////////////////////////////////////////////////
    // add
    ///////////////////////////////////////////////////////////////////////////
    add4 #(
        .input_len     (input_len)
        ) addtree0 (
        .clk           (clk),
        .rst_n         (rst_n),
        .data_in1      (data_in_internal[0]),
        .data_in2      (data_in_internal[1]),
        .data_in3      (data_in_internal[2]),
        .data_in4      (data_in_internal[3]),
        .data_out      (data_io0_internal)
    );

    add4 #(
        .input_len     (input_len)
        ) addtree1 (
        .clk           (clk),
        .rst_n         (rst_n),
        .data_in1      (data_in_internal[4]),
        .data_in2      (data_in_internal[5]),
        .data_in3      (data_in_internal[6]),
        .data_in4      (data_in_internal[7]),
        .data_out      (data_io1_internal)
    );

    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
            data_out_internal <= {(input_len+4){1'b0}};
        else
            data_out_internal <= (data_io0_internal + data_io1_internal) + data_in_internal8;
    end

    // output
    assign data_out = data_out_internal;

    ///////////////////////////////////////////////////////////////////////////
    //
    ///////////////////////////////////////////////////////////////////////////

endmodule

module add10 #(
    parameter input_len     = 16
    ) (
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire [input_len * 10 - 1 : 0]    data_in,   // input1
    output wire [input_len + 4 - 1 : 0]     data_out
    );

    ///////////////////////////////////////////////////////////////////////////
    // internal signal definition
    ///////////////////////////////////////////////////////////////////////////

    wire signed     [input_len - 1 : 0]   data_in_internal[10 - 1 : 0];
    wire signed     [input_len + 1 : 0]   data_io0_internal;
    wire signed     [input_len + 1 : 0]   data_io1_internal;

    reg  signed     [input_len - 1 : 0]   data_in_internal8;
    reg  signed     [input_len - 1 : 0]   data_in_internal9;

    reg  signed     [input_len + 3 : 0]   data_out_internal;
    genvar i;

    ///////////////////////////////////////////////////////////////////////////
    // input
    ///////////////////////////////////////////////////////////////////////////

    generate
    for(i=0; i<10; i=i+1)
    begin
        assign data_in_internal[i] = data_in[input_len * i + input_len - 1: input_len * i];
    end
    endgenerate

    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
            data_in_internal8 <= 0;
        else
            data_in_internal8 <= data_in_internal[8];
    end

    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
            data_in_internal9 <= 0;
        else
            data_in_internal9 <= data_in_internal[9];
    end

    ///////////////////////////////////////////////////////////////////////////
    // add
    ///////////////////////////////////////////////////////////////////////////
    add4 #(
        .input_len     (input_len)
        ) addtree0 (
        .clk           (clk),
        .rst_n         (rst_n),
        .data_in1      (data_in_internal[0]),
        .data_in2      (data_in_internal[1]),
        .data_in3      (data_in_internal[2]),
        .data_in4      (data_in_internal[3]),
        .data_out      (data_io0_internal)
    );

    add4 #(
        .input_len     (input_len)
        ) addtree1 (
        .clk           (clk),
        .rst_n         (rst_n),
        .data_in1      (data_in_internal[4]),
        .data_in2      (data_in_internal[5]),
        .data_in3      (data_in_internal[6]),
        .data_in4      (data_in_internal[7]),
        .data_out      (data_io1_internal)
    );

    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
            data_out_internal <= 0;
        else
            data_out_internal <= (data_io0_internal + data_io1_internal) + (data_in_internal8 + data_in_internal9);
    end

    // output
    assign data_out = data_out_internal;

    ///////////////////////////////////////////////////////////////////////////
    //
    ///////////////////////////////////////////////////////////////////////////

endmodule