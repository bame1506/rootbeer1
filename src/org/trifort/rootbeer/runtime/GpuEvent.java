package org.trifort.rootbeer.runtime;

import java.util.List;

/* event implementation storing the data for sharing during exchange or
 * parallel coordination of an event.
 * @see https://lmax-exchange.github.io/disruptor/docs/com/lmax/disruptor/EventFactory.html */
import com.lmax.disruptor.EventFactory;

/**
 * This class wraps GpuFuture with an additional GPU event command and a
 * kernel list which are accessible in parallel by using EventFactory.
 */
public class GpuEvent
{
    private GpuEventCommand m_value;
    private List<Kernel>    m_work;
    private final GpuFuture m_future; /**<- object to signal if concurrent thread has finished */

    public GpuEvent() { m_future = new GpuFuture(); }
    /* Access methods boilerplate code */
    public GpuEventCommand getValue()                         { return m_value ; }
    public GpuFuture       getFuture()                        { return m_future; }
    public List<Kernel>    getKernelList()                    { return m_work  ; }
    public void            setValue( GpuEventCommand value )  { m_value = value; }
    public void            setKernelList( List<Kernel> work ) { m_work  = work;  }

    /* @todo: I don't even understand what this is? It looks like a mixture of
     * a member variable and a method ... */
    public final static EventFactory<GpuEvent> EVENT_FACTORY = new EventFactory<GpuEvent>()
    {
        public GpuEvent newInstance() { return new GpuEvent(); }
    };
}
