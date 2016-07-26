/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.deadmethods;


import java.util.List;
import java.util.Set;

import org.trifort.rootbeer.util.ReadFile;


public class DeadMethods
{
    private static final boolean debugging = false;

    private List<Block> m_blocks;
    private Set<String> m_live;

    /**
     * Reads CUDA source code file into String and call parseString on it
     */
    public void parseFile( String filename )
    {
        ReadFile reader = new ReadFile( filename );
        String contents = "";
        try {
            contents = reader.read();
        }
        catch ( Exception ex )
        {
            ex.printStackTrace(System.out);
        }

        parseString(contents);
    }

    public void parseString( String contents )
    {
        /* categorize lines by comment, string, ... */
        final List<Segment> segments     = new SegmentParser().parse( contents );
        /* group segments to method declarations, method bodies, ... */
        final List<Block>   blocks       = new BlockParser().parse( segments );
        /* get the method names and save them into each block with Block.setMethod */
        final List<String>  method_names = new MethodNameParser().parse( blocks );
        /* find and set the names of methods each block actually invokes */
        new MethodAnnotator().parse( blocks, method_names );

        m_blocks = blocks;

        /* debug output */
        if ( debugging )
        {
            System.out.println( "+------------ DeadMethods.parseString ------------" );
            /**
             * Sample Output:
             * | TYPE_FREE   : __device__  char *
             * | TYPE_FREE   : org_trifort_gc_deref( int handle )
             * | TYPE_FREE   : {
             * | TYPE_FREE   :     char * data_arr = (char *) m_Local[0];
             * | TYPE_COMMENT: // dpObjectMem
             * | TYPE_FREE   :     long long lhandle = handle;
             * | TYPE_FREE   :     lhandle = lhandle << 4;
             * | TYPE_FREE   :     return &data_arr[lhandle];
             * | TYPE_FREE   : }
             * I.e. segments are functional units of some kind, often this
             * corresponds to one segment per line, but things like __device__
             * and end-of-line comments are split into another segment
             *
             * Uncomment this to see if SegmentParser works correctly
             */
            for ( final Segment segment : segments )
            {
                System.out.print( "| TYPE_" );
                switch ( segment.getType() )
                {
                    case SegmentParser.TYPE_FREE   : System.out.print( "FREE   " ); break;
                    case SegmentParser.TYPE_COMMENT: System.out.print( "COMMENT" ); break;
                    case SegmentParser.TYPE_STRING : System.out.print( "STRING " ); break;
                    case SegmentParser.TYPE_CHAR   : System.out.print( "CHAR   " ); break;
                    case SegmentParser.TYPE_DEFINE : System.out.print( "DEFINE " ); break;
                    default:
                        throw new RuntimeException( "unknown type: " +
                        segment.getType() + " str: " + segment.getString() );
                }
                System.out.println( ": " + segment.getString() );
            }

            /**
             * Test if the function name and body and function calls were
             * recognized correctly
             *
             * Example Output:
             * @verbatim
             *   | <block method=no, name="">
             *   | #include <assert.h>
             *   | </block>
             *   | <block method=no, name="">
             *   | __constant__ size_t m_Local[3];
             *   | </block>
             *   | <block method=yes, name="getThreadId">
             *   | __device__ int getThreadId()
             *   {
             *       int linearized = 0;
             *       int max        = 1;
             *
             *       linearized += threadIdx.z * max; max *= blockDim.z;
             *       linearized += threadIdx.y * max; max *= blockDim.y;
             *       linearized += threadIdx.x * max; max *= blockDim.x;
             *       linearized += blockIdx.y  * max; max *= gridDim.y;
             *       linearized += blockIdx.x  * max;
             *       return linearized;
             *   }| </block>
             *   | <block method=yes, name="getThreadIdxx">
             *   | __device__ int       getThreadIdxx(){
             *    return threadIdx.x; }| </block>
             * @endverbatim
             */
            for ( final Block block : blocks )
            {
                System.out.print( "| <block method=" );
                System.out.print( block.isMethod() ? "yes" : "no" );
                System.out.print( ", name=\"" );
                if ( block.isMethod() )
                    System.out.print( block.getMethod().getName() );
                System.out.println( "\">" );
                System.out.println( "| " + block.toString() );
                if ( block.isMethod() )
                {
                    final List<String> invokedNames = block.getMethod().getInvoked();
                    System.out.print( "| invoked these " + invokedNames.size() + " methods: " );
                    for ( final String invoked : invokedNames )
                        System.out.print( invoked + " " );
                    System.out.println();
                }
                System.out.println( "| </block>" );
            }
        }
    }

    /**
     * Returns a newline separated list of methods which are in the set 'live'
     * and also of other non-method members.
     * This basically returns the source code which is actually needed.
     */
    private String outputLive
    (
        final List<Block> blocks,
        final Set<String> live
    )
    {
        StringBuilder ret = new StringBuilder();
        for ( Block block : blocks )
        {
            if ( block.isMethod() )
            {
                /* @todo isn't this possibly hazardous to use contains instead
                 * of equals? */
                if ( ! live.contains( block.getMethod().getName() ) )
                {
                    if ( debugging )
                        System.out.println( "[DeadMethods.java] Filtered out dead method '" + block.getMethod().getName() + "'" );
                    continue;
                }
                ret.append( block.toString() );
                ret.append( "\n" );
            }
            else
            {
                ret.append( block.toString() );
                ret.append( "\n" );
            }
        }

        if ( debugging )
        {
            //System.out.println( "+------------ DeadMethods.outputLive ------------" );
            //System.out.println( ret.toString() );
        }

        return ret.toString();
    }

    public String getResult()
    {
        if ( m_live == null )
            m_live = new LiveMethodDetector().parse( m_blocks );
        return outputLive( m_blocks, m_live );
    }

    public String getCompressedResult()
    {
        if ( m_live == null )
            m_live = new LiveMethodDetector().parse( m_blocks );
        List<Block> blocks = new MethodNameCompressor().parse( m_blocks, m_live );
        return outputLive(blocks, m_live);
    }
}
