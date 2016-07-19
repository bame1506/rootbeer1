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
    private static final boolean debugging = true;

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
        final SegmentParser segment_parser = new SegmentParser();
        final List<Segment> segments       = segment_parser.parse(contents);

        final BlockParser block_parser     = new BlockParser();
        final List<Block> blocks           = block_parser.parse(segments);

        final MethodNameParser name_parser = new MethodNameParser();
        final List<String> method_names    = name_parser.parse(blocks);

        final MethodAnnotator annotator    = new MethodAnnotator();
        annotator.parse( blocks, method_names );

        m_blocks = blocks;

        /* debug output */
        if ( debugging )
        {
            System.out.println( "+------------ DeadMethods.parseString ------------" );
            for ( Segment segment : segments )
            {
                System.out.println( "| " + segment.toString() );
            }
            for ( Block block : blocks )
            {
                System.out.println( "| <block>" );
                System.out.println( "| " + block.toString() );
                System.out.println( "| </block>" );
            }
            for ( Block block : blocks )
            {
                if ( block.isMethod() )
                {
                    Method method = block.getMethod();
                    System.out.println( "| " + method.getName() );
                }
            }
            for ( Block block : blocks )
            {
                if ( block.isMethod() )
                {
                    Method method = block.getMethod();
                    System.out.println( "| name: "+method.getName() );
                    for ( String invoked : method.getInvoked() )
                        System.out.println( "|   invoked: "+invoked );
                }
            }
        }
    }

    /**
     * Returns a newline separated list of methods which are in the set live
     * and also of other non-method members
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
                    continue;
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
            System.out.println( "+------------ DeadMethods.outputLive ------------" );
            System.out.println( ret.toString() );
        }

        return ret.toString();
    }

    public String getResult()
    {
        if ( m_live == null )
        {
            LiveMethodDetector detector = new LiveMethodDetector();
            m_live = detector.parse( m_blocks );
        }

        return outputLive( m_blocks, m_live );
    }

    public String getCompressedResult()
    {
        if ( m_live == null )
        {
            LiveMethodDetector detector = new LiveMethodDetector();
            m_live = detector.parse(m_blocks);
        }

        MethodNameCompressor compressor = new MethodNameCompressor();
        List<Block> blocks = compressor.parse(m_blocks, m_live);

        return outputLive(blocks, m_live);
    }
}
