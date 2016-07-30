/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime;


import java.util.Map;
import java.util.TreeMap;


/**
 * Mainly these are the raw function onto which, using Soot
 * @see rootbeer/entry/CompilerSetup.java , CUDA intrinsics are mapped to
 * Therefore I think this library somehow also gets compiled using Soot and
 * Jimple.
 */
public class RootbeerGpu
{
    /* @todo as these are static I hope there is some kind of synchronize when
     * writing to them? */
    private static boolean isOnGpu   ;
    private static byte[]  sharedMem ;
    private static int     threadIdxx;
    private static int     threadIdxy;
    private static int     threadIdxz;
    private static int     blockDimx ;
    private static int     blockDimy ;
    private static int     blockDimz ;
    private static int     blockIdxx ;
    private static int     blockIdxy ;
    private static long    gridDimx  ;
    private static long    gridDimy  ;

    private static Map<Integer, Object> m_sharedArrayMap;

    static
    {
        isOnGpu          = false;
        sharedMem        = new byte[48*1024];
        m_sharedArrayMap = new TreeMap<Integer, Object>();
    }

    public static boolean isOnGpu(){ return isOnGpu; }
    public static void setIsOnGpu(boolean value){ isOnGpu = value; }

    /**
     * @return blockIdx.x * blockDim.x + threadIdx.x;
     */
    public static int getThreadId()
    {
        int blockSize = blockDimx * blockDimy * blockDimz;

        int ret = (blockIdxx  * (int) gridDimy * blockSize) +
                  (blockIdxy  * blockSize) +
                  (threadIdxx * blockDimy * blockDimz) +
                  (threadIdxy * blockDimz) +
                  (threadIdxz);
        return ret;
    }

    public static int  getThreadIdxx(){ return threadIdxx; }
    public static int  getThreadIdxy(){ return threadIdxy; }
    public static int  getThreadIdxz(){ return threadIdxz; }
    public static int  getBlockIdxx (){ return blockIdxx ; }
    public static int  getBlockIdxy (){ return blockIdxy ; }
    public static int  getBlockDimx (){ return blockDimx ; }
    public static int  getBlockDimy (){ return blockDimy ; }
    public static int  getBlockDimz (){ return blockDimz ; }
    public static long getGridDimx  (){ return gridDimx  ; }
    public static long getGridDimy  (){ return gridDimy  ; }

    /* I'm not sure if these should be public! I only see it used in tests,
     * so there is no use or conversion to these in using Soot, unlike for
     * the getters.
     * I only see these used in one tests. I think they only exist for that
     * one test which is bad programming style, especially without comments
     */
    public static void setThreadIdxx(int  thread_idxx){ threadIdxx = thread_idxx; }
    public static void setThreadIdxy(int  thread_idxy){ threadIdxy = thread_idxy; }
    public static void setThreadIdxz(int  thread_idxz){ threadIdxz = thread_idxz; }
    public static void setBlockIdxx (int  block_idxx ){ blockIdxx  = block_idxx ; }
    public static void setBlockIdxy (int  block_idxy ){ blockIdxy  = block_idxy ; }
    public static void setBlockDimx (int  block_dimx ){ blockDimx  = block_dimx ; }
    public static void setBlockDimy (int  block_dimy ){ blockDimy  = block_dimy ; }
    public static void setBlockDimz (int  block_dimz ){ blockDimz  = block_dimz ; }
    public static void setGridDimx  (long grid_dimx  ){ gridDimx   = grid_dimx  ; }
    public static void setGridDimy  (long grid_dimy  ){ gridDimy   = grid_dimy  ; }

    public static int  syncthreadsCount(int count){ return 0; }
    public static void syncthreads      (){}
    public static void threadfence      (){}
    public static void threadfenceBlock (){}
    public static void threadfenceSystem(){}

    /* @todo these look wrong! They either do nothing or return bogus.
     *       they don't seem to be used either -> commented out */
    //public static long getRef(Object obj){ return 0; }
    //public static Object getSharedObject(int index){ return null; }
    //public static void setSharedObject  (int index, Object value){}

    public static byte getSharedByte(int index){ return sharedMem[index]; }
    public static void setSharedByte(int index, byte value){ sharedMem[index] = value; }
    public static boolean getSharedBoolean(int index){ return sharedMem[index] == 1 ? true : false; }
    public static void setSharedBoolean(int index, boolean value){ sharedMem[index] = (byte)( value ? 1 : 0 ); }

    public static char getSharedChar(int index)
    {
        char ret = 0;
        ret |=  sharedMem[index + 0]       & 0x00ff;
        ret |= (sharedMem[index + 1] << 8) & 0xff00;
        return ret;
    }

    public static void setSharedChar(int index, char value)
    {
        sharedMem[index + 0] = (byte)( value       & 0xff);
        sharedMem[index + 1] = (byte)((value >> 8) & 0xff);
    }

    public static short getSharedShort(int index)
    {
        /* @todo could be done in loop -> private method ? */
        short ret = 0;
        ret |=  sharedMem[index + 0]       & 0x00ff;
        ret |= (sharedMem[index + 1] << 8) & 0xff00;
        return ret;
    }

    public static void setSharedShort(int index, short value)
    {
        /* @todo could be done in loop -> private method ? */
        sharedMem[index + 0] = (byte)(  value       & 0xff );
        sharedMem[index + 1] = (byte)( (value >> 8) & 0xff );
    }

    public static int getSharedInteger(int index)
    {
        /* @todo could be done in loop */
        int ret = 0;
        ret |=  sharedMem[index + 0]        & 0x000000ff;
        ret |= (sharedMem[index + 1] <<  8) & 0x0000ff00;
        ret |= (sharedMem[index + 2] << 16) & 0x00ff0000;
        ret |= (sharedMem[index + 3] << 24) & 0xff000000;
        return ret;
    }

    public static void setSharedInteger(int index, int value)
    {
        /* @todo could be done in loop */
        sharedMem[index + 0] = (byte) (  value        & 0xff );
        sharedMem[index + 1] = (byte) ( (value >> 8 ) & 0xff );
        sharedMem[index + 2] = (byte) ( (value >> 16) & 0xff );
        sharedMem[index + 3] = (byte) ( (value >> 24) & 0xff );
    }

    public static long getSharedLong(int index)
    {
        long ret = 0;
        /* @todo could be done in loop */
        ret |=  (long) sharedMem[index + 0]        & 0x00000000000000ffL;
        ret |= ((long) sharedMem[index + 1] <<  8) & 0x000000000000ff00L;
        ret |= ((long) sharedMem[index + 2] << 16) & 0x0000000000ff0000L;
        ret |= ((long) sharedMem[index + 3] << 24) & 0x00000000ff000000L;
        ret |= ((long) sharedMem[index + 4] << 32) & 0x000000ff00000000L;
        ret |= ((long) sharedMem[index + 5] << 40) & 0x0000ff0000000000L;
        ret |= ((long) sharedMem[index + 6] << 48) & 0x00ff000000000000L;
        ret |= ((long) sharedMem[index + 7] << 56) & 0xff00000000000000L;
        return ret;
    }

    public static void setSharedLong(int index, long value)
    {
        /* @todo could be done in loop */
        sharedMem[index + 0] = (byte) ( value        & 0xff);
        sharedMem[index + 1] = (byte) ((value >> 8 ) & 0xff);
        sharedMem[index + 2] = (byte) ((value >> 16) & 0xff);
        sharedMem[index + 3] = (byte) ((value >> 24) & 0xff);
        sharedMem[index + 4] = (byte) ((value >> 32) & 0xff);
        sharedMem[index + 5] = (byte) ((value >> 40) & 0xff);
        sharedMem[index + 6] = (byte) ((value >> 48) & 0xff);
        sharedMem[index + 7] = (byte) ((value >> 56) & 0xff);
    }

    public static float  getSharedFloat (int i){ return Float.intBitsToFloat( getSharedInteger(i) ); }
    public static double getSharedDouble(int i){ return Double.longBitsToDouble( getSharedLong(i) ); }
    public static void   setSharedFloat (int i, float  v){ setSharedInteger( i, Float.floatToIntBits(v) ); }
    public static void   setSharedDouble(int i, double v){ setSharedLong( i, Double.doubleToLongBits(v) ); }

    /* @todo his looks very wrong !!! so I commented it out */
    //public static double sin(double value){ return 0; }

    public static void   atomicAddGlobal (int[]    a, int i, int    v){ synchronized(a){ a[i] += v; } }
    public static void   atomicAddGlobal (long[]   a, int i, long   v){ synchronized(a){ a[i] += v; } }
    public static void   atomicAddGlobal (float[]  a, int i, float  v){ synchronized(a){ a[i] += v; } }
    public static double atomicAddGlobal (double[] a, int i, double v){ synchronized(a){ double old = a[i]; a[i] += v; return old; } }
    public static int    atomicSubGlobal (int[]    a, int i, int    v){ synchronized(a){ int    old = a[i]; a[i] -= v; return old; } }
    public static int    atomicExchGlobal(int[]    a, int i, int    v){ synchronized(a){ int    old = a[i]; a[i]  = v; return old; } }
    public static long   atomicExchGlobal(long[]   a, int i, long   v){ synchronized(a){ long   old = a[i]; a[i]  = v; return old; } }
    public static float  atomicExchGlobal(float[]  a, int i, float  v){ synchronized(a){ float  old = a[i]; a[i]  = v; return old; } }
    public static int atomicAndGlobal    (int[]    a, int i, int    v){ synchronized(a){ int    old = a[i]; a[i] &= v; return old; } }
    public static int atomicOrGlobal     (int[]    a, int i, int    v){ synchronized(a){ int    old = a[i]; a[i] |= v; return old; } }
    public static int atomicXorGlobal    (int[]    a, int i, int    v){ synchronized(a){ int    old = a[i]; a[i] ^= v; return old; } }
    public static int atomicMinGlobal    (int[]    a, int i, int    v){ synchronized(a){ int    old = a[i]; if(v < old){ a[i] = v; } return old; } }
    public static int atomicMaxGlobal    (int[]    a, int i, int    v){ synchronized(a){ int    old = a[i]; if(v > old){ a[i] = v; } return old; } }
    public static int atomicCASGlobal(int[] a, int i, int compare, int v){ synchronized(a){ int old = a[i]; if(old == compare){ a[i] = v; } return old; } }

    /*
    //TODO: working on this
    public static int[] createSharedIntArray(int index, int length){
        int[] ret = new int[length];
        m_sharedArrayMap.put(index, ret);
        return ret;
    }

    public static int[] getSharedIntArray(int index){
        if(m_sharedArrayMap.containsKey(index)){
            return (int[]) m_sharedArrayMap.get(index);
        } else {
            throw new IllegalArgumentException();
        }
    }
    */
}
