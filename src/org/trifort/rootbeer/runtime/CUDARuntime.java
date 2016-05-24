package org.trifort.rootbeer.runtime;

import java.util.List;

/**
 * Class which buffers GPU list returned by cudaGetDeviceProperties native
 */
public class CUDARuntime implements IRuntime
{
    private final List<GpuDevice> m_cards;

    public CUDARuntime(){ m_cards = loadGpuDevices(); }
    @Override public List<GpuDevice> getGpuDevices() { return m_cards; }

    private native List<GpuDevice> loadGpuDevices();
}
