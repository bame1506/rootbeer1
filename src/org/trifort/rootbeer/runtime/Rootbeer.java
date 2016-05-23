package org.trifort.rootbeer.runtime;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;


public class Rootbeer
{
    private IRuntime m_cudaRuntime;
    private List<GpuDevice> m_cards;

    /* static constructor which extracts the CUDA shared libraries */
    static {
        CUDALoader loader = new CUDALoader();
        loader.load();
    }

    public Rootbeer(){}

    public List<GpuDevice> getDevices()
    {
        if ( m_cards != null ) {
            return m_cards;
        }

        m_cards = new ArrayList<GpuDevice>();
        try {
          Class c = Class.forName("org.trifort.rootbeer.runtime.CUDARuntime");
          Constructor<IRuntime> ctor = c.getConstructor();
          m_cudaRuntime = ctor.newInstance();
          m_cards.addAll(m_cudaRuntime.getGpuDevices());
        } catch(Exception ex){
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

    public Context createDefaultContext(){
      List<GpuDevice> devices = getDevices();
      GpuDevice best = null;
      for(GpuDevice device : devices){
        if(best == null){
          best = device;
        } else {
          if(device.getMultiProcessorCount() > best.getMultiProcessorCount()){
            best = device;
          }
        }
      }
      if(best == null){
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
    public ThreadConfig getThreadConfig( List<Kernel> kernels, GpuDevice device )
    {
        BlockShaper block_shaper = new BlockShaper();
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

    public void run(List<Kernel> work)
    {
        Context context = createDefaultContext();
        ThreadConfig thread_config = getThreadConfig(work, context.getDevice());
        try {
            context.setThreadConfig(thread_config);
            context.setKernel(work.get(0));
            context.setUsingHandles(true);
            context.buildState(); // this sets the GPU to use
            context.run(work);
        } finally {
            context.close();
        }
    }
}
