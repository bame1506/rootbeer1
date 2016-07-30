/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.bytecode;


public class Constants
{
    //if SiceGcInfo is 16, the synch tests fail
    public final static int SizeGcInfo          = 32;
    public final static int ArrayOffsetSize     = 32;
    /* @see README.md in section Developing.
     * These are used for the compression of heap addresses
     * Don't change these values! Some places are still hardcoded with 4 bit
     * or respectively 16 Byte alignemnt!
     */
    public final static int MallocAlignZeroBits = 4;
    public final static int MallocAlignBytes    = 1 << MallocAlignZeroBits;
}
