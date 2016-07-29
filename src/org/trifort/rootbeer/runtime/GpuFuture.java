package org.trifort.rootbeer.runtime;


import org.trifort.rootbeer.runtimegpu.GpuException;


/**
 * This class does nothing more than providing a variable 'm_ready' which can be
 * set by a thread and read by another to signal if the thread has finished.
 */
public class GpuFuture
{
    /* These variables are used to communicate between threads, therefore they
     * must be declared volatile */
    private volatile boolean   m_ready;
    private volatile Throwable m_ex   ;

    /* Access methods */
    public GpuFuture()   { m_ready = false; }
    public void signal() { m_ready = true ; }
    public void reset()  { m_ready = false; m_ex = null; }
    public void setException( Exception ex ) { m_ex = ex; } /* used by onEvent */

    /**
     * Waits for task (what task?) to finish
     *
     * This take is not to be confused with that of e.g. LinkedBlockingQueue
     * which doesn't synchronize if called.
     */
    public void take()
    {
        /* @todo: an atomic lock would be better here, because in principal
         * it could happen, that the other read sets read to true and then
         * again to false! */
        while( ! m_ready )
        {
            /* Wait for volatile variable to be set to true. Don't do a
             * busy-loop to save CPU usage */
            try {
                java.lang.Thread.sleep( 50 /* ms */ );
            } catch( Exception m_ex ) {
                // doesn't matter
            }
        }
        /* If the worker thread has received an exception, then also throw it
         * on this thread */
        if ( m_ex != null )
        {
            /* This code just casts the most general exception "Throwable"
             * to more special ones to make it more meaningfull. Is there
             * not a function which already does this somehow? */
            if( m_ex instanceof NullPointerException )
                throw (NullPointerException) m_ex;
            else if( m_ex instanceof OutOfMemoryError )
                throw (OutOfMemoryError) m_ex;
            else if( m_ex instanceof Error )
                throw (Error) m_ex;
            else if( m_ex instanceof ArrayIndexOutOfBoundsException )
                throw (ArrayIndexOutOfBoundsException) m_ex;
            else if( m_ex instanceof RuntimeException )
                throw (RuntimeException) m_ex;
            else
                throw new RuntimeException( m_ex );
        }
    }

}
