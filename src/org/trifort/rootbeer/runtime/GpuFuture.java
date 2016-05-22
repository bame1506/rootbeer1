package org.trifort.rootbeer.runtime;

import org.trifort.rootbeer.runtimegpu.GpuException;

/**
 * This class does nothing more than providing a variable 'ready' which can be
 * set by a thread and read by another to signal if the thread has finished.
 */
public class GpuFuture
{
    /* These variables are used to communicate between threads, therefore they
     * must be declared volatile */
    private volatile boolean   ready;
    private volatile Throwable ex;

    public GpuFuture(){
        ready = false;
    }

    public void signal() {
        ready = true;
    }

    public void reset() {
        ex = null;
        ready = false;
    }

    /**
     * Waits for task (what task?) to finish
     *
     * This take is not to be confused with that of e.g. LinkedBlockingQueue
     * which doesn't synchronize if called.
     */
    public void take()
    {
        while(!ready){
          //do nothing
        }
        if(ex != null)
        {
            /* This code just casts possibly derived exceptions to more
             * general ones, maybe to simplify error handling?
             * @todo: seems useless to me. A calling function may catch
             *        these exceptions similarly with instanceof theirselves */
            if(ex instanceof NullPointerException){
                throw (NullPointerException) ex;
            } else if(ex instanceof OutOfMemoryError){
                throw (OutOfMemoryError) ex;
            } else if(ex instanceof Error){
                throw (Error) ex;
            } else if(ex instanceof ArrayIndexOutOfBoundsException){
                throw (ArrayIndexOutOfBoundsException) ex;
            } else if(ex instanceof RuntimeException){
                throw (RuntimeException) ex;
            } else {
                throw new RuntimeException(ex);
            }
        }
    }

    public void setException(Exception ex) {
      this.ex = ex;
    }
}
