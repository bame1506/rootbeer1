/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime;


import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;

import org.trifort.rootbeer.generate.bytecode.Constants;


public abstract class Serializer
{
    public final Memory mMem;
    public final Memory mTextureMem;

    private final static Map<Object, Long>  mWriteToGpuCache;
    private final static Map<Long, Object>  mReverseWriteToGpuCache;
    private final static Map<Long, Object>  mReadFromGpuCache;
    private final static Map<Long, Integer> m_classRefToTypeNumber;

    /* static initializer for static members */
    static
    {
        mWriteToGpuCache        = new IdentityHashMap<Object, Long>();
        mReverseWriteToGpuCache = new HashMap<Long, Object >();
        mReadFromGpuCache       = new HashMap<Long, Object >();
        m_classRefToTypeNumber  = new HashMap<Long, Integer>();
    }

    public Serializer( final Memory mem, final Memory texture_mem )
    {
        if ( mem == null )
            throw new IllegalArgumentException( "[Serializer.java] Argument 'mem' = null is not allowed!" );
        if ( texture_mem == null )
            throw new IllegalArgumentException( "[Serializer.java] Argument 'texture_mem' = null is not allowed!" );

        mMem        = mem;
        mTextureMem = texture_mem;
        mReadFromGpuCache      .clear();
        mWriteToGpuCache       .clear();
        mReverseWriteToGpuCache.clear();
        m_classRefToTypeNumber .clear();
    }

    /* default argument: write_data = true */
    public long writeToHeap( final Object o ) { return writeToHeap(o, true); }
    public void writeStaticsToHeap() { doWriteStaticsToHeap(); }

    public void addClassRef(long ref, int class_number){ m_classRefToTypeNumber.put(ref, class_number); }

    public int[] getClassRefArray()
    {
        int max_type = 0;
        for ( final int num : m_classRefToTypeNumber.values() )
            if ( num > max_type ) max_type = num;

        int[] ret = new int[ max_type+1 ];
        for ( final long compressedAddress : m_classRefToTypeNumber.keySet() )
        {
            final int pos = m_classRefToTypeNumber.get( compressedAddress );
            assert( pos < max_type + 1 );
            ret[pos] = (int)( compressedAddress >> Constants.MallocAlignZeroBits );
        }
        return ret;
    }

    private static class WriteCacheResult
    {
        public long m_Ref;
        public boolean m_NeedToWrite;
        public WriteCacheResult( long ref, boolean need_to_write )
        {
            m_Ref = ref;
            m_NeedToWrite = need_to_write;
        }
    }

    private static synchronized WriteCacheResult checkWriteCache
    (
        final Object  o        ,
        final int     size     ,
        final boolean read_only, // unused ???
        final Memory  mem
    )
    {
        //strings are cached in Java 1.6, we need to make strings individual units
        //for rootbeer so concurrent modifications change different objects
        if ( o instanceof String )
        {
            long ref = mem.mallocWithSize( size );
            return new WriteCacheResult( ref, true );
        }
        else
        {
            if ( mWriteToGpuCache.containsKey(o) )
            {
                long ref = mWriteToGpuCache.get(o);
                return new WriteCacheResult(ref, false);
            }
            long ref = mem.mallocWithSize(size);
            mWriteToGpuCache.put(o, ref);
            mReverseWriteToGpuCache.put(ref, o);
            return new WriteCacheResult(ref, true);
        }
    }

    public long writeToHeap( Object o, boolean write_data )
    {
        if ( o == null )
            throw new IllegalArgumentException( "[Serializer.java:writeToHeap] Argument 'o' = null is not allowed!" );

        /* not sure if this is a good idea, because calling functions only rarely check */
        //if ( o == null )
        //    return -1;

        final int size = doGetSize(o);
        final boolean read_only = false;
        WriteCacheResult result = checkWriteCache(o, size, read_only, mMem);;

        if ( ! result.m_NeedToWrite )
            return result.m_Ref;
        //if(o == null){
        //  System.out.println("writeToHeap: null at addr: "+result.m_Ref);
        //} else {
        //  System.out.println("writeToHeap: "+o.toString()+" at addr: "+result.m_Ref);
        //}
        doWriteToHeap(o, write_data, result.m_Ref, read_only);
        //BufferPrinter printer = new BufferPrinter();
        //printer.print(mMem, result.m_Ref, 128);
        return result.m_Ref;
    }

    /**
     * A call to this is generated in generate/bytecode/VisitorReadGen.java
     */
    protected Object checkCache( final long address, final Object item )
    {
        /* synchronized: in the context of multithreading execute this block
         * atomically in order to avoid race conditions. */
        synchronized( mReadFromGpuCache )
        {
            if ( mReadFromGpuCache.containsKey( address ) )
                return mReadFromGpuCache.get(address);
            else
                mReadFromGpuCache.put( address, item );
                return item;
        }
    }

    /**
     * @param[in] o I think this is only needed for the type, not the actual
     *              object which most likely will be null anyways?
     *              @todo I'm not sure, but I think the object is unchanged
     *              by this method
     */
    public Object readFromHeap
    (
        final Object  o,
        final boolean read_data,
        final long    address
    )
    {
        /* Note that the address is assumed to have gone through writeRef
         * followed by readRef. As those functions for some reason store
         * the actual address in increments of 16, the address will actually
         * be:
         *   address = ( originalAddress / 16 ) * 16  => -1 -> 0
         *   address = ( originalAddress >> 4 ) << 4  => -1 -> -16
         * As I didn't know of this damn check I sometimes changed the bitshift
         * to / 16 and * 16. But that is not fully reversible. E.g. for -1.
         * That's why a bitshift was used. But the other way around I DID
         * add checks, e.g.: if ( value % 16 != 0 ) throw Exception;
         *                   else value /= 16;
         * @todo The check down below might actually work, but only if
         * null_ptr_check is cast to int! else it should be compared with -16!
         */
        long null_ptr_check = address >> Constants.MallocAlignZeroBits;
        if ( null_ptr_check == -1 )
            return null;

        synchronized ( mReadFromGpuCache )
        {
            if ( mReadFromGpuCache.containsKey(address) )
            {
                Object ret = mReadFromGpuCache.get(address);
                return ret;
            }
        }

        //if(o == null){
        //    System.out.println("readFromHeap: null. addr: "+address);
        //} else {
        //    System.out.println("readFromHeap: "+o.toString()+". addr: "+address+" class: "+o.getClass());
        //}
        //BufferPrinter printer = new BufferPrinter();
        //printer.print(mMem, address-128, 256);
        Object ret = doReadFromHeap( o, read_data, address );
        //if(ret == null){
        //    System.out.println("doReadFromHeap: null. addr: "+address);
        //} else {
        //    if(ret instanceof char[]){
        //        char[] char_array = (char[]) ret;
        //        String str = new String(char_array);
        //        System.out.println("doReadFromHeap char[]: "+str+".["+char_array.length+"] addr: "+address);
        //    } else {
        //        System.out.println("doReadFromHeap: "+ret.toString()+". addr: "+address);
        //    }
        //}
        return ret;
    }

    public void readStaticsFromHeap(){ doReadStaticsFromHeap(); }

    private Object readField( Class cls, Object base, String name )
    {
        while ( true )
        {
            try
            {
                Field f = cls.getDeclaredField( name );
                f.setAccessible( true );
                Object ret = f.get( base );
                return ret;
            }
            catch ( Exception ex )
            {
                cls = cls.getSuperclass();
                //java.lang.Throwable.backtrace cannot be found this way, I don't know why.
                if ( cls == null ) {
                    return null;
                }
            }
        }
    }
    public Object readField ( Object base, String name ) { return readField( base.getClass(), base, name ); }
    public Object readStaticField( Class cls, String name ) { return readField( cls, null, name ); }

    /**
     * Tries to reflectively set a possibly static field for a class.
     * If no field with the name exist, tries superclass.
     */
    private void writeField
    (
        Class        cls  ,
        final Object base ,
        final String name ,
        final Object value
    )
    {
        while ( true )
        {
            try
            {
                Field f = cls.getDeclaredField( name );
                f.setAccessible( true );
                f.set ( base, value );
                return;
            }
            catch(Exception ex)
            {
                cls = cls.getSuperclass();
            }
        }
    }
    public void writeField( Object base, String name, Object value ) { writeField( base.getClass(), base, name, value ); }
    public void writeStaticField       (Class cls, String name, Object  value){ writeField( cls, null, name, value ); }
    public void writeStaticByteField   (Class cls, String name, byte    value){ writeStaticField(cls,name,value); }
    public void writeStaticBooleanField(Class cls, String name, boolean value){ writeStaticField(cls,name,value); }
    public void writeStaticCharField   (Class cls, String name, char    value){ writeStaticField(cls,name,value); }
    public void writeStaticShortField  (Class cls, String name, short   value){ writeStaticField(cls,name,value); }
    public void writeStaticIntField    (Class cls, String name, int     value){ writeStaticField(cls,name,value); }
    public void writeStaticLongField   (Class cls, String name, long    value){ writeStaticField(cls,name,value); }
    public void writeStaticFloatField  (Class cls, String name, float   value){ writeStaticField(cls,name,value); }
    public void writeStaticDoubleField (Class cls, String name, double  value){ writeStaticField(cls,name,value); }

    /* defined by VisitorWriteGenStatic.java using soot.
     * I don't exactly see why it is necessary to use Soot to write these out
     * I guess because Soot is needed to analyze all members of a class and
     * then write them out
     * @todo do these methods change the internal pointer of mMem i.e.
     *       m_ObjectMemory ???
     */
    public abstract void   doWriteToHeap (Object o, boolean write_data, long ref, boolean read_only);
    public abstract Object doReadFromHeap(Object o, boolean read_data , long ref);
    public abstract void   doWriteStaticsToHeap();
    public abstract void   doReadStaticsFromHeap();
    public abstract int    doGetSize(Object o);
}
