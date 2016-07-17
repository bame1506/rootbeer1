/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime;

import java.util.ArrayList;
import java.util.List;

public class BlockShaper
{
    private int m_GridShape ;
    private int m_BlockShape;

    private int ceilDiv(
        final int a,
        final int b
    )
    {
        return ( a + b - 1 ) / b;
    }

    /**
     * Chooses an a kernel configuration for a given number of parallelism
     *
     * E.g. the GTX 760 can and should run at most 12288 threads in parallel,
     * although it has only 384 real cores 1152 CUDA cores (6 Kepler-SMX with
     * each 192 CUDA Cores and each task scheduler can handle 2048 threads),
     * in order to make full use of pipelining.
     */
    public void run
    (
        final int       num_items,
        final GpuDevice gpu
    )
    {
        final int nMaxThreads = 256; // 128 or 512 are also OK values

        m_BlockShape = nMaxThreads;
        m_GridShape  = ceilDiv( num_items, nMaxThreads );

        assert( m_GridShape  > 0 );
        assert( m_BlockShape > 0 );
        assert( m_GridShape * m_BlockShape <= num_items );
    }

    public int gridShape () { return m_GridShape ; }
    public int blockShape() { return m_BlockShape; }
}
