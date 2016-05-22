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
    private GpuEventCommand value;
    private List<Kernel>    work;
    private final GpuFuture future; /**<- object to signal if concurrent thread has finished */

    public GpuEvent(){
        future = new GpuFuture();
    }

    public GpuEventCommand getValue() {
        return value;
    }

    public GpuFuture getFuture(){
        return future;
    }

    public void setValue(GpuEventCommand value) {
        this.value = value;
    }

    public void setKernelList(List<Kernel> work) {
        this.work = work;
    }

    public List<Kernel> getKernelList(){
        return work;
    }

    /* @todo: I don't even understand what this is? It looks like a mixture of
     * a member variable and a method ... */
    public final static EventFactory<GpuEvent> EVENT_FACTORY = new EventFactory<GpuEvent>()
    {
        public GpuEvent newInstance() {
            return new GpuEvent();
        }
    };
}
