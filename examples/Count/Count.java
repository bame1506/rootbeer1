
import java.io.*;          // System.out.println
import java.util.Arrays;
import java.util.Scanner;  // nextLong
import java.util.ArrayList;
import java.util.List;
import java.lang.Long;

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.GpuDevice;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.Context;

public class Count
{

    public static void main ( String[] args )
    {
        final Rootbeer  rootbeerContext = new Rootbeer();
        final GpuDevice device          = rootbeerContext.getDevices().get(0);

        final int nKernels = device.getMultiProcessorCount()
                           * device.getMaxThreadsPerMultiprocessor();

        final long nRollsPerThreads = 1684701;

        long[] nHitsA = new long[nKernels];
        long[] nHitsB = new long[nKernels];
        List<Kernel> tasks = new ArrayList<Kernel>();
        for ( int i = 0; i < nKernels; ++i )
        {
            /* Note that nHitsA[i] + nHitsB[i] should be 0 for the test
             * after the kernel run */
            nHitsA[i] =  i;
            nHitsB[i] = -i;
            tasks.add( new CountKernel( nHitsA, nHitsB, nRollsPerThreads ) );
        }

        final Context context = device.createContext( -1  /* auto choose memory size. Not sure what this is about or what units -.- */ );
        final int threadsPerBlock = 256;
        ThreadConfig thread_config = new ThreadConfig(
            threadsPerBlock, /* threadCountX */
            1, /* threadCountY */
            1, /* threadCountZ */
            ( tasks.size() + threadsPerBlock - 1 ) / threadsPerBlock, /* blockCountX */
            1, /* blockCountY */
            tasks.size() /* numThreads */
        );

        try
        {
            context.setThreadConfig( thread_config );
            context.setKernel( tasks.get(0) );
            context.setUsingHandles( true ); /* ? */
            context.buildState();
            System.out.println( "[Count.java:main] Calling context.run( tasks )" );
            context.run( tasks );
        }
        finally
        {
            context.close();
        }

        /* Kernel finished, now print */
        for ( int i = 0; i < nKernels; ++i )
        {
            boolean theyMatchUp = nHitsA[i] + nHitsB[i] == nRollsPerThreads;
            if ( ( ! theyMatchUp ) ||  i < 10 )
            {
                System.out.println( "" + nHitsA[i] + " inside " +
                                         nHitsB[i] + " outside" );
                if ( ! theyMatchUp )
                {
                    System.out.println( "  => Something is wrong! Don't add up to " +
                                        nRollsPerThreads );
                }
            }
        }
    }
}
