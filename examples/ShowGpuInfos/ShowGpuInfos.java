
import java.io.*;
import java.lang.Long;  // MAX_VALUE
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.GpuDevice;
import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.ThreadConfig;

import org.trifort.rootbeer.generate.opencl.tweaks.GencodeOptions;
import org.trifort.rootbeer.util.CudaPath;

public class ShowGpuInfos
{
    static public void gpuDeviceInfo( int riDevice )
    {
        long t0, t1;
        t0 = System.nanoTime();
        Rootbeer rootbeerContext = new Rootbeer();
        t1 = System.nanoTime();
        System.out.println( "Creating Rootbeer context took " + ((t1-t0)/1e9) + " seconds" );

        t0 = System.nanoTime();
        List<GpuDevice> devices = rootbeerContext.getDevices();
        if ( riDevice >= devices.size() )
            throw new IllegalArgumentException(
                "Only " + devices.size() + " graphic cards could be found. " +
                "The specified device ID " + riDevice + " is therefore not available!"
            );
        GpuDevice device = devices.get( riDevice );

        System.out.println( "\n================== Device Number " + device.getDeviceId() + " ==================" );
        System.out.println( "| Device name              : " + device.getDeviceName() );
        System.out.println( "| Device type              : " + device.getDeviceType() );
        System.out.println( "| Computability            : " + device.getMajorVersion() + "." + device.getMinorVersion() );
		System.out.println( "|------------------- Architecture -------------------" );
        System.out.println( "| Number of SM             : " + device.getMultiProcessorCount() );
        System.out.println( "| Max Threads per SM       : " + device.getMaxThreadsPerMultiprocessor() );
        System.out.println( "| Max Threads per Block    : " + device.getMaxThreadsPerBlock() );
        System.out.println( "| Warp Size                : " + device.getWarpSize() );
        System.out.println( "| Clock Rate               : " + (device.getClockRateHz() / 1e6) + " GHz" );
        System.out.println( "| Max Block Size           : (" + device.getMaxBlockDimX() + "," + device.getMaxBlockDimY() + "," + device.getMaxBlockDimZ() + ")" );
        System.out.println( "| Max Grid Size            : (" + device.getMaxGridDimX() + "," + device.getMaxGridDimY() + "," + device.getMaxGridDimZ() + ")" );
		System.out.println( "|---------------------- Memory ----------------------" );
        System.out.println( "| Total Global Memory      : " + device.getTotalGlobalMemoryBytes() + " Bytes" );
        System.out.println( "| Free Global Memory       : " + device.getFreeGlobalMemoryBytes() + " Bytes" );
        System.out.println( "| Total Constant Memory    : " + device.getTotalConstantMemoryBytes() + " Bytes" );
        System.out.println( "| Shared Memory per Block  : " + device.getMaxSharedMemoryPerBlock() + " Bytes" );
        System.out.println( "| Registers per Block      : " + device.getMaxRegistersPerBlock() );
        System.out.println( "| Memory Clock Rate        : " + (device.getMemoryClockRateHz() / 1e9 ) + " GHz" );
        System.out.println( "| Memory Pitch             : " + device.getMaxPitch() );
        System.out.println( "| Device is Integrated     : " + device.getIntegrated() );
        System.out.println( "=====================================================" );
        t1 = System.nanoTime();
        System.out.println( "Getting GPU devince information took " + ((t1-t0)/1e9) + " seconds" );
    }

    static public void main( String[] args )
    {
        /* Test some other metric gathering from Rootbeer */
        System.out.println( "CUDA path found   : " + CudaPath.get() );
        System.out.println( "NVCC version found: " + GencodeOptions.getNVCCVersion() );

        /* List all GPUs */
        for ( int i = 0; i < 256; ++i )
        {
            try
            {
                gpuDeviceInfo(i);
            }
            catch ( IllegalArgumentException e )
            {
                break ;
            }
        }
    }

}
