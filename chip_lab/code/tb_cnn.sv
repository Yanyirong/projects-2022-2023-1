// --------------------------------------------------------------------------
//
// Module Info: Testbench top-level
// Language   : System{Verilog}
// Description:
//		- Provide the input image data
//		- Provide the input Filter data
//		- Load the output CNN data into the file
//
// --------------------------------------------------------------------------

`timescale 1ns/1ps
module tb_cnn;
    reg				clk;
    reg				rst;
    reg				image_ready;
    //== Image interface == //
    reg	[7:0]		image_mem[0:4355];	//image size: 64 x 64 =4096
    wire [12:0] 	image_addr; 
    wire	 		image_rden;
    reg [7:0] 	    image_i;
    reg			    image_valid;
    //== Fitler interface == //
    reg [7:0]		filter_mem[0:8];	//Filter 3x3x3
    wire [3:0] 		filter_addr;
    wire			filter_rden;
    reg [7:0] 	    filter_i;
    reg			    filter_valid;
    int 			fp_out; 
    logic 			[7:0]  cnn_data_o; 
    reg 			[15:0] cnt_dout;

    //----------------------clock and reset--------
    initial
    begin
        clk = 1'b0;
        rst = 1'b0;
        image_ready = 1'b0;
        image_valid = 1'b0;
        image_i = 8'd0;
        filter_valid = 1'b0;
        filter_i = 8'd0;
        #100ns rst = 1'b1;
        #100ns rst = 1'b0;
        
        $readmemh("image_data.txt", image_mem); 
        $readmemh("weight_data.txt", filter_mem);
                $display("/n##### @%0t, Data is ready ####/n", $time);  

        #1000ns image_ready = 1'b1;    
        #10ns 	image_ready = 1'b0; 
    end

    // 100MHz clock
    always 
    begin
        #5ns clk = ~clk;
    end

    //#################----Input Interface----##################
    always @ (posedge clk)
    if (image_rden == 1'b1)
    begin
        image_i <= image_mem[image_addr];
        image_valid <= 1'b1;
    end
    else
    begin
        image_valid <= 1'b0;
    end

    always @ (posedge clk)
    if (filter_rden == 1'b1)
    begin
        filter_i <= filter_mem[filter_addr];
        filter_valid <= 1'b1;
    end
    else
    begin
        filter_valid <= 1'b0;
    end
      
    // -----------------------------------------
    // DUT
    // -----------------------------------------
    cnn dut(
    //------------ Image Reshaper ------------------------
        .clk		                (clk      	),
        .rst_n                      (!rst          	),
        //Input interface
        .image_ready				(image_ready	),
        .image_rden_o  				(image_rden 	),
        .image_addr_o  				(image_addr 	),
        .image_i     				(image_i    	),
        .image_valid				(image_valid	),
        //Fiter input	
        .filter_rden_o 				(filter_rden  	),
        .filter_addr_o 				(filter_addr  	),
        .filter_i    				(filter_i     	),
        .filter_valid				(filter_valid	),
        //CNN output 	
        .cnn_valid_o 				(cnn_valid_o  	),
        .cnn_data_o  				(cnn_data_o   	)
    );

//============ Output Interface =============//


initial begin
    fp_out = $fopen("cnn_output.txt", "w");	
    cnt_dout = 16'd0;
end
    //Load output into data file   
    always @ (posedge clk) begin
        if( cnn_valid_o == 1'b1 ) begin
            $fdisplay(fp_out, "%h", cnn_data_o );
        end
    end
    //Debug for valid output data
    always @ (posedge clk) begin
        if (cnn_valid_o == 1'b1 )
            cnt_dout <= cnt_dout + 1'b1;
    end
    
//Calculate total latency
reg [20:0] 	latency;
reg [1:0]  	latency_ena;
wire		conv_done;



initial 
begin
    latency = 16'd0;
end
    always @ (posedge clk) 
    begin
        if ( cnt_dout == 16'd4096 ) 
            latency_ena[0] <= 1'b0;
        else if (image_ready == 1'b1 )
            latency_ena[0] <= 1'b1;
    end

    always @ (posedge clk) 
    begin			
        latency_ena[1] <= latency_ena[0];
    end

    assign conv_done = (latency_ena[1:0] == 2'b10);

    always @ (posedge clk) 
    begin
        if (latency_ena[0] == 1'b1 )
            latency <= latency + 1'b1;
    end


    always @ (posedge clk) 
    begin
        if (cnt_dout >= 16'd4097)
        begin
            $display("/n##### @%0t, Output data number is over 4096 ####/n",$time ); 
        end
        else if ( conv_done == 1'b1 )
        begin
            $display("/n##### Latency   =",latency," cycles ####/n"); 
            $stop();
        end
    end
endmodule
