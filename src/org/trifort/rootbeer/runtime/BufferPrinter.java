/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime;


import java.util.ArrayList;
import java.util.List;


/**
 * Just a debugging class which prints a hexdump of memory like objects
 * to stdout
 */
public class BufferPrinter
{
    /**
     * Prints a hexdump of the memory 16 values per line to stdout
     */
    public static void print( final Memory mem, long start_ptr, long length )
    {
        /* backup old mem pointer */
        if ( start_ptr < 0 )
            throw new RuntimeException( "[BufferPrinter.java:print] Invalid argument " +
                start_ptr + " given for start pointer!" );
        if ( length < 1 )
            length = mem.getSize();
        final long pointerBackup = mem.getPointer();
        mem.setAddress( start_ptr );

        final int nBytesPerLine  = 16;
        final int nBytesPerGroup =  4;
        final int addressPadding = ( "" + ( start_ptr + length ) ).length();

        /* make ArrayList of Strings. Each String corresponding to one line,
         * e.g.: 00 00 00 00  00 00 00 00  00 00 00 00  00 00 00 00 */
        for ( int iByte = 1; iByte <= length; ++iByte )
        {
            /* print address at beginning of line */
            if ( (iByte-1) % nBytesPerLine == 0 )
                System.out.print( String.format( "%0" + addressPadding + "d : ",
                                                 start_ptr + iByte - 1 ) );

            System.out.print( String.format( "%02x", mem.readByte() ) + " " );
            if ( iByte % nBytesPerLine == 0 )
                System.out.println();
            else if ( iByte % nBytesPerGroup == 0 )
                System.out.print( " " );
        }

        /* restore old address */
        mem.setAddress( pointerBackup );
    }
}
