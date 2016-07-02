/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime;

public class ThreadConfig
{
    private int m_threadCountX;
    private int m_threadCountY;
    private int m_threadCountZ;
    private int m_blockCountX ;
    private int m_blockCountY ;
    private int m_numThreads  ;

    public ThreadConfig
    (
        final int threadCountX,
        final int threadCountY,
        final int threadCountZ,
        final int blockCountX,
        final int blockCountY,
        final int numThreads
    )
    {
        m_threadCountX = threadCountX;
        m_threadCountY = threadCountY;
        m_threadCountZ = threadCountZ;
        m_blockCountX  = blockCountX ;
        m_blockCountY  = blockCountY ;
        m_numThreads   = numThreads  ;
        assert( m_threadCountX >= 1 );
        assert( m_threadCountY >= 1 );
        assert( m_threadCountZ >= 1 );
        assert( m_blockCountX  >= 1 );
        assert( m_blockCountY  >= 1 );
        assert( m_numThreads == m_threadCountX *
                                m_threadCountY *
                                m_threadCountZ *
                                m_blockCountX  *
                                m_blockCountY );
    }

    public int getThreadCountX(){ return m_threadCountX; }
    public int getThreadCountY(){ return m_threadCountY; }
    public int getThreadCountZ(){ return m_threadCountZ; }
    public int getBlockCountX (){ return m_blockCountX ; }
    public int getBlockCountY (){ return m_blockCountY ; }
    public int getNumThreads  (){ return m_numThreads  ; }
}
