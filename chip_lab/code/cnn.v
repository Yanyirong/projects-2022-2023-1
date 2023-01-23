module cnn (
        input               clk                 ,
        input               rst_n               ,
        //Input image interface                 
        input               image_ready         ,
        output reg          image_rden_o        ,
        output reg [12:0]   image_addr_o        ,
        input [7:0]         image_i             ,
        input               image_valid         ,
        //Fiter input                           
        output              filter_rden_o       ,
        output reg [3:0]    filter_addr_o       ,
        input [7:0]         filter_i            ,
        input               filter_valid        ,
        //CNN output                            
        output reg          cnn_valid_o         ,
        output reg [7:0]    cnn_data_o          
    );

    //----------------------Insert your design code here--------------------------------
    ///////////////////////////////////////////////////////////////////////////
    // internal parameter and variables definition
    ///////////////////////////////////////////////////////////////////////////
    localparam Y                  = 66; 
    localparam X                  = 66; 

    localparam input_width        = 8;
    localparam weight_width       = 8;
    localparam output_width       = 8;

    localparam WAIT_READY         = 3'b000;
    localparam INIT_WGT           = 3'b001;
    localparam INIT_IMG           = 3'b011;
    localparam COMPUTE            = 3'b010;
    localparam FINISH             = 3'b110;

    reg  [2:0]                      state_c;
    reg  [2:0]                      state_n;

    wire                            rd_en;
    reg  [7 - 1                : 0] x_count;
    reg  [7 - 1                : 0] y_count;
    reg  [3 * input_width  - 1 : 0] image_in;
    reg                             input_last;
    reg  [weight_width - 1     : 0] wgt_reg[9 - 1 : 0];
    wire [input_width - 1      : 0] sram_din[3 : 0];
    wire [7 - 1                : 0] sram_addr[3 : 0];
    wire [3                    : 0] sram_we;
    wire [input_width - 1      : 0] sram_dout[3 : 0];
    genvar i;

    ////////////////////////////////////////////////////////////
    // counter
    ////////////////////////////////////////////////////////////
    always@(posedge clk or negedge rst_n)
    begin
        if(!rst_n) 
            y_count <= 0;
        else 
        if((state_c[1] == 1'b1) & (image_valid | (state_c == FINISH)))
        begin
            if(y_count == Y - 1)
                y_count <= 0;
            else
                y_count <= y_count + 1;
        end
    end

    always@(posedge clk or negedge rst_n)
    begin
        if(!rst_n) 
            x_count <= 0;
        else 
        if((state_c[1] == 1'b1) & image_valid & (y_count == Y - 1))
        begin
            if(x_count == X + 1)
                x_count <= 0;
            else
                x_count <= x_count + 1;
        end
    end

    always@(posedge clk or negedge rst_n)
    begin
        if(!rst_n) 
            {wgt_reg[0], wgt_reg[1], wgt_reg[2], wgt_reg[3], wgt_reg[4], wgt_reg[5], wgt_reg[6], wgt_reg[7], wgt_reg[8]} <= 0;
        else 
            if(filter_valid)
                wgt_reg[filter_addr_o - 1] <= filter_i;
    end

    always@(posedge clk or negedge rst_n) 
    begin
        if (!rst_n)
            input_last <= 0;
        else
            input_last <= rd_en & (y_count == Y - 2);
    end
    
    ////////////////////////////////////////////////////////////
    // addr_out
    ////////////////////////////////////////////////////////////
    
    always@(posedge clk or negedge rst_n) 
    begin
        if (!rst_n)
            image_rden_o <= 0;
        else
            if(state_c[1]== 1'b1)
                image_rden_o <= 1'b1;
            else
                image_rden_o <= 0;
    end

    always@(posedge clk or negedge rst_n) 
    begin
        if (!rst_n)
            image_addr_o <= 0;
        else
            if(state_c[1]== 1'b1)
                image_addr_o <= image_addr_o + 1;
    end

    always@(posedge clk or negedge rst_n) 
    begin
        if (!rst_n)
            filter_addr_o <= 0;
        else
        begin
            if(state_c == INIT_WGT)
                filter_addr_o <= filter_addr_o + 1;
        end
    end

    assign filter_rden_o = (state_c == INIT_WGT);
    ////////////////////////////////////////////////////////////
    // finite state machine
    ////////////////////////////////////////////////////////////
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n) 
            state_c <= WAIT_READY;
        else 
            state_c <= state_n;
    end

    always @(*)
    begin
        case (state_c)
            WAIT_READY:
                if(image_ready)
                    state_n = INIT_WGT;

            INIT_WGT: 
                if(filter_addr_o == 9 - 1) 
                    state_n = INIT_IMG;
                else
                    state_n = INIT_WGT;

            INIT_IMG: if(x_count > 2) state_n = COMPUTE;

            COMPUTE: 
                if ((y_count == Y - 1) & (x_count == X - 1)) 
                    state_n = FINISH;

            FINISH:  
                if ((y_count == Y - 1) & (x_count == X + 1)) 
                    state_n = WAIT_READY;

            default: state_n = WAIT_READY;
        endcase
    end

    assign rd_en = ((state_n == COMPUTE) | (state_n == FINISH)) & (x_count > 2);

    ////////////////////////////////////////////////////////////
    // weight
    ////////////////////////////////////////////////////////////
    wire kernel_valid;
    reg [3 * input_width - 1 : 0]  kernel_in;

    assign  kernel_valid = (state_n == COMPUTE) && (y_count < 3);
    always@(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
            kernel_in <= 0;
        else
        begin
            if(kernel_valid & (y_count == 7'd0))
                kernel_in <= {wgt_reg[6], wgt_reg[3], wgt_reg[0]};
            else if(kernel_valid & (y_count == 7'd1))
                kernel_in <= {wgt_reg[7], wgt_reg[4], wgt_reg[1]};
            else if(kernel_valid & (y_count == 7'd2))
                kernel_in <= {wgt_reg[8], wgt_reg[5], wgt_reg[2]};
        end
    end

    ////////////////////////////////////////////////////////////
    // line_buffer
    ////////////////////////////////////////////////////////////
    reg [input_width - 1 : 0] image_reg;

    always@(posedge clk or negedge rst_n)
    begin
        if(!rst_n) 
            image_reg <= 0;
        else 
            image_reg <= image_i;
    end

    generate
    for(i=0; i<4; i=i+1)
    begin
        assign sram_din[i] = (x_count[1:0] == i) ? image_reg : 0;
        assign sram_addr[i] = y_count;
        assign sram_we[i] = (x_count[1:0] == i) ? (state_c[1] == 1'b1) & image_valid : 1'b0;
    
        line_buffer#(
            .addr_width     (7),
            .data_width     (8),
            .data_depth     (128)
        ) line_buffer(
            .clka           (clk),
            .ena            (1'b1),
            .wena           (sram_we[i]),
            .addra          (sram_addr[i]),

            .dina           (sram_din[i]),
            .douta          (sram_dout[i])
        );
    end
    endgenerate

    ////////////////////////////////////////////////////////////
    // Dconv_3x3_PE
    ////////////////////////////////////////////////////////////
    wire                            output_valid;
    wire [output_width - 1 : 0]     data_out;
    reg                             PE_valid;
    reg                             kernel_valid_reg;
    reg  [2 - 1 : 0]                sram_count_reg;

    always@(posedge clk or negedge rst_n)
    begin
        if(!rst_n) 
            sram_count_reg <= 0;
        else 
            sram_count_reg <= x_count[1:0];
    end

    always@(posedge clk or negedge rst_n)
    begin
        if(!rst_n) 
            PE_valid <= 0;
        else 
            PE_valid <= rd_en;
    end

    always@(posedge clk or negedge rst_n)
    begin
        if(!rst_n) 
            kernel_valid_reg <= 0;
        else 
            kernel_valid_reg <= kernel_valid;
    end

    always @(*) 
    begin
        case (sram_count_reg)
            2'd3: image_in = {sram_dout[2], sram_dout[1], sram_dout[0]};
            2'd0: image_in = {sram_dout[3], sram_dout[2], sram_dout[1]};
            2'd1: image_in = {sram_dout[0], sram_dout[3], sram_dout[2]};
            2'd2: image_in = {sram_dout[1], sram_dout[0], sram_dout[3]};
            default: image_in = {sram_dout[2], sram_dout[1], sram_dout[0]};
        endcase  
    end

    Dconv_3x3_PE Dconv_3x3_PE(
        .clk            (clk),
        .rst_n          (rst_n),
        .kernel         (kernel_in),
        .image          (image_in),

        .image_valid    (PE_valid),
        .kernel_valid   (kernel_valid_reg),
        .input_last     (input_last),

        .data_out       (data_out),
        .output_valid   (output_valid)
    );

    always@(posedge clk or negedge rst_n)
    begin
        if(!rst_n) 
            cnn_data_o <= 0;
        else 
            cnn_data_o <= data_out;
    end

    always@(posedge clk or negedge rst_n)
    begin
        if(!rst_n) 
            cnn_valid_o <= 0;
        else 
            cnn_valid_o <= output_valid & (x_count > 2);
    end
endmodule