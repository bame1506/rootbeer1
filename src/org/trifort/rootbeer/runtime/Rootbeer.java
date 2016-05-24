package org.trifort.rootbeer.runtime;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;


public class Rootbeer
{
    private IRuntime m_cudaRuntime;
    private List<GpuDevice> m_cards;

    /* static constructor which extracts and loads the CUDA shared libraries */
    static {
        CUDALoader loader = new CUDALoader();
        loader.load();
    }

    public Rootbeer(){}

    /**
     * On first call this function creates a CUDARuntime.java instance
     * and returns all found GPUs
     * @see CUDARuntime.getGpuDevices
     */
    public List<GpuDevice> getDevices()
    {
        if ( m_cards != null ) {
            return m_cards;
        }

        m_cards = new ArrayList<GpuDevice>();
        try
        {
            /* @todo: Why not just use: m_cudaRuntime = new CUDARuntime() ? */
            final Class c = Class.forName( "org.trifort.rootbeer.runtime.CUDARuntime" );
            Constructor<IRuntime> ctor = c.getConstructor();
            m_cudaRuntime = ctor.newInstance();
            m_cards.addAll( m_cudaRuntime.getGpuDevices()) ;
        }
        catch ( Exception ex )
        {
            ex.printStackTrace();
            //ignore
        }

        //if(m_cards.isEmpty()){
        //  try {
        //    Class c = Class.forName("org.trifort.rootbeer.runtime.OpenCLRuntime");
        //    Constructor<IRuntime> ctor = c.getConstructor();
        //    m_openCLRuntime = ctor.newInstance();
        //    m_cards.addAll(m_openCLRuntime.getGpuDevices());
        //  } catch(Exception ex){
        //    //ignore
        //  }
        //}

        return m_cards;
    }

    /**
     * Chooses best (number of CUDA cores) GPU from list.
     *
     * When using Multi-GPU user can copy-paste parts of this method.
     * @todo calculate peak flops instead of naive CUDA core count.
     */
    public Context createDefaultContext()
    {
        final List<GpuDevice> devices = getDevices();
        if ( devices.size() <= 0 ) {
            throw new java.lang.RuntimeException( "No CUDA-available devices found!" );
        }
        GpuDevice best = devices.get(0);
        for ( GpuDevice device : devices.subList( 1, devices.size() ) )
        {
            if ( device.getPeakFlops() > best.getPeakFlops() ) {
                best = device;
            }
        }
        if ( best == null ) {
            return null;
        } else {
            return best.createContext();
        }
    }

    /**
     * automatically determines a good kernel configuration based on device
     * properties for a given number of threads wanted
     * @param[in] kernels Only kernels.size will be evaluated
     **/
    public ThreadConfig getThreadConfig( final List<Kernel> kernels, final GpuDevice device )
    {
        final BlockShaper block_shaper = new BlockShaper();
        block_shaper.run( kernels.size(), device.getMultiProcessorCount() );

        return new ThreadConfig(
            block_shaper.blockShape(), /* threadCountX */
            1,                         /* threadCountY */
            1,                         /* threadCountZ */
            block_shaper.gridShape(),  /* blockCountX  */
            1,                         /* blockCountY  */
            kernels.size()             /* numThreads   */
        );
    }

    public void run( final List<Kernel> work )
    {
        Context context = createDefaultContext();
        ThreadConfig thread_config = getThreadConfig( work, context.getDevice() );
        try {
            context.setThreadConfig(thread_config);
            context.setKernel( work.get(0) );
            context.setUsingHandles( true );
            context.buildState();   // this sets the GPU to use and all other parameters
            context.run( work );
        } finally {
            context.close();
        }
    }
}
