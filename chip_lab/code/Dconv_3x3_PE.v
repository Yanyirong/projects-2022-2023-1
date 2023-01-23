//////////////////////////////////////////////////////////////////////////////////
// Company:        SJTU
// Engineer:       Jinming Zhang
// Create Date:    10:30 06/26/2021 
// Design Name:    
// Module Name:    Dconv_3x3_PE 
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
module Dconv_3x3_PE#(
    parameter input_width   = 8,
    parameter weight_width  = 8,
    parameter output_width  = 8
    )( 
    input wire clk,
    input wire rst_n,  
    input wire [3 * weight_width - 1 : 0] kernel,
    input wire [3 * input_width  - 1 : 0] image,

    input wire image_valid,
    input wire kernel_valid,
    input wire input_last,

    output wire [output_width - 1 : 0] data_out,
    output reg  output_valid
    );
    
    ///////////////////////////////////////////////////////////////////////////
    // internal parameter and variables definition
    ///////////////////////////////////////////////////////////////////////////
    localparam mul_out_width        = input_width + weight_width; // 16
    localparam add_out_width        = mul_out_width + 4; // 16 + 4 = 20

    reg  [9 * weight_width - 1     : 0]  kernel_reg;
    reg  [9 * input_width  - 1     : 0]  image_reg;
    reg  [6 - 1                    : 0]  valid_reg;
    reg  [2 - 1                    : 0]  state_c;
    reg  [2 - 1                    : 0]  state_n;
    reg  [weight_width      - 1    : 0]  kernel_mul[9 - 1 : 0];
    reg  [input_width       - 1    : 0]  image_mul[9 - 1 : 0];
    wire [mul_out_width     - 1    : 0]  mul_out[9 - 1 : 0];
    wire [add_out_width - 1 : 0]         add_out;

    genvar i, ch;

    ///////////////////////////////////////////////////////////////////////////
    // state
    ///////////////////////////////////////////////////////////////////////////
    localparam INIT_IMG = 2'b00;
    localparam COMPUTE = 2'b01;
    localparam FINISH = 2'b11;

    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n) 
            state_c <= INIT_IMG;
        else 
            state_c <= state_n;
    end

    always @(*)
    begin
        case (state_c)
            INIT_IMG: 
                if(image_valid) 
                    state_n = COMPUTE;
                else
                    state_n = INIT_IMG;

            COMPUTE:  
                if (input_last)
                    state_n = FINISH;
                else 
                    state_n = COMPUTE;

            FINISH:  state_n = INIT_IMG;
            default: state_n = INIT_IMG;
        endcase
    end

    always@(posedge clk or negedge rst_n) 
    begin
        if (!rst_n)
        begin
            valid_reg <= 0;
            output_valid <= 0;
        end 
        else
        begin
            {output_valid, valid_reg} <= {valid_reg, (state_c == COMPUTE)}; 
        end
    end

    ///////////////////////////////////////////////////////////////////////////
    // kernel_reg 
    ///////////////////////////////////////////////////////////////////////////
    always@(posedge clk or negedge rst_n)
    begin
        if (!rst_n)
        begin
            kernel_reg[(0 + 1) * weight_width - 1 : 0 * weight_width] <= 0;
            kernel_reg[(1 + 1) * weight_width - 1 : 1 * weight_width] <= 0;
            kernel_reg[(2 + 1) * weight_width - 1 : 2 * weight_width] <= 0;
        end
            
        else
        begin
            kernel_reg[(0 + 1) * weight_width - 1 : 0 * weight_width] <= (kernel_valid) ? kernel[weight_width * 1 - 1 : weight_width * 0 + 0] : kernel_reg[(0 + 1) * weight_width - 1 : 0 * weight_width];

            kernel_reg[(1 + 1) * weight_width - 1 : 1 * weight_width] <= (kernel_valid) ? kernel[weight_width * 2 - 1 : weight_width * 1 + 0] : kernel_reg[(1 + 1) * weight_width - 1 : 1 * weight_width];

            kernel_reg[(2 + 1) * weight_width - 1 : 2 * weight_width] <= (kernel_valid) ? kernel[weight_width * 3 - 1 : weight_width * 2 + 0] : kernel_reg[(2 + 1) * weight_width - 1 : 2 * weight_width];
        end
    end

    always@(posedge clk or negedge rst_n)
    begin
        if (!rst_n)
        begin
            kernel_reg[6 * input_width - 1 : 3 * input_width] <= 0;
            kernel_reg[9 * input_width - 1 : 6 * input_width] <= 0;
        end
        else
        begin
            kernel_reg[6 * input_width - 1 : 3 * input_width] <= (kernel_valid) ? kernel_reg[3 * input_width - 1 : 0 * input_width] : kernel_reg[6 * input_width - 1 : 3 * input_width];

            kernel_reg[9 * input_width - 1 : 6 * input_width] <= (kernel_valid) ? kernel_reg[6 * input_width - 1 : 3 * input_width] : kernel_reg[9 * input_width - 1 : 6 * input_width];
        end
    end

    ///////////////////////////////////////////////////////////////////////////
    // image_reg 
    ///////////////////////////////////////////////////////////////////////////
    always@(posedge clk or negedge rst_n)
    begin
        if (!rst_n)
        begin
            image_reg[(0 + 1) * input_width - 1 : 0 * input_width] <= 0;
            image_reg[(1 + 1) * input_width - 1 : 1 * input_width] <= 0;
            image_reg[(2 + 1) * input_width - 1 : 2 * input_width] <= 0;
        end
        else
        begin
            image_reg[(0 + 1) * input_width - 1 : 0 * input_width] <= (image_valid) ? image[input_width * 1 - 1 : input_width * 0 + 0] : image_reg[(0 + 1) * input_width - 1 : 0 * input_width];

            image_reg[(1 + 1) * input_width - 1 : 1 * input_width] <= (image_valid) ? image[input_width * 2 - 1 : input_width * 1 + 0] : image_reg[(1 + 1) * input_width - 1 : 1 * input_width];

            image_reg[(2 + 1) * input_width - 1 : 2 * input_width] <= (image_valid) ? image[input_width * 3 - 1 : input_width * 2 + 0] : image_reg[(2 + 1) * input_width - 1 : 2 * input_width];
        end
    end

    always@(posedge clk or negedge rst_n)
    begin
        if (!rst_n)
        begin
            image_reg[6 * input_width - 1 : 3 * input_width] <= 0;
            image_reg[9 * input_width - 1 : 6 * input_width] <= 0;
        end
        else
        begin
            image_reg[6 * input_width - 1 : 3 * input_width] <= (image_valid) ? image_reg[3 * input_width - 1 : 0  * input_width] : image_reg[6 * input_width - 1 : 3 * input_width];

            image_reg[9 * input_width - 1 : 6 * input_width] <= (image_valid) ? image_reg[6 * input_width - 1 : 3 * input_width] : image_reg[9 * input_width - 1 : 6 * input_width];
        end
    end

    ///////////////////////////////////////////////////////////////////////////
    // mul
    ///////////////////////////////////////////////////////////////////////////
    generate
    for(i=0; i<9; i=i+1)
    begin
        always@(posedge clk or negedge rst_n)
        begin
            if (!rst_n)
            begin
                kernel_mul[i] <= 0;
                image_mul[i] <= 0;
            end
            else
            begin
                kernel_mul[i] <= kernel_reg[(i + 1) * input_width - 1 : i * input_width];
                image_mul[i]  <= image_reg [(i + 1) * input_width - 1 : i * input_width];
            end
        end

        mul #(
            .input_width   (input_width),
            .weight_width  (weight_width),
            .mul_out_width (mul_out_width)
            ) mul (
            .clk           (clk),
            .rst_n         (rst_n),
            .data1         (kernel_mul[i]),
            .data2         (image_mul[i]),
            .result        (mul_out[i])
        );
    end
    endgenerate

    ///////////////////////////////////////////////////////////////////////////
    // add9
    ///////////////////////////////////////////////////////////////////////////
    add9 #(
        .input_len (mul_out_width)
    ) add9 (
        .clk           (clk),
        .rst_n         (rst_n),
        .data_in       ({mul_out[8], mul_out[7], mul_out[6], mul_out[5], mul_out[4], mul_out[3], mul_out[2], mul_out[1], mul_out[0]}),
        .data_out      (add_out)
    );

    ///////////////////////////////////////////////////////////////////////////
    // cutoff
    ///////////////////////////////////////////////////////////////////////////
    cutoff #(
        .input_width        (add_out_width),
        .output_width       (output_width),
        .radix_point_right  ()
    ) cutoff (
        .clk           (clk),
        .rst_n         (rst_n),
        .data_in       (add_out),
        .data_out      (data_out)
    );

endmodule