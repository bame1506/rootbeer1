/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.deadmethods;


import java.util.ArrayList;
import java.util.List;


public class BlockParser
{
    public static final int TYPE_FREE    = 0;
    public static final int TYPE_DEFINE  = 1;
    public static final int TYPE_DECLARE = 2;
    public static final int TYPE_METHOD  = 3;

    /**
     * Collects possibly multiple segments together to one block
     * E.g. a define block or function block beginning with the function
     * header and ending with a corresponding closing curly bracket
     */
    public List<Block> parse( final List<Segment> segments )
    {
        final List<Block> ret = new ArrayList<Block>();
        for ( int i = 0; i < segments.size(); ++i )
        {
            final Segment segment = segments.get(i);

            /* This only adds preprocessor commands on the top-level to a new
             * block. not those inside functions */
            if ( segment.getType() == SegmentParser.TYPE_DEFINE )
            {
                ret.add( new Block( segment, TYPE_DEFINE ) );
                continue;
            }

            /* Ignore comments on the top level. Note that there shouldn't
             * possibly be char or string without a TYPE_FREE first! */
            if ( segment.getType() == SegmentParser.TYPE_STRING ||
                 segment.getType() == SegmentParser.TYPE_CHAR     )
            {
                throw new RuntimeException( "Found a free standing character or string ("
                    + segment.getString() + "). This is not allowed." );
            }
            /* ignore comments at top level */
            if ( segment.getType() == SegmentParser.TYPE_COMMENT )
                continue;

            /* TYPE_FREE basically means code */
            if ( segment.getType() != SegmentParser.TYPE_FREE )
                throw new RuntimeException( "Segment '" + segment.getString() + "'" +
                    " is of unkown type: " + segment.getType() );

            /*********** work on TYPE_FREE segment ***********/
            final String str = segment.getString().trim();
            if ( str.isEmpty() )
                continue;

            final char last_char = str.charAt ( str.length()  - 1 );
            /* because of how the SegmentParser works there is now curly
             * bracket inside the string if there is none at the last place.
             * For the first TYPE_FREE this means it is a declaration.
             * After that we ensure that we are at the top level again,
             * before working on the next segment, meaning it must be
             * a declaration again
             */
            if ( last_char == ';' )
            {
                ret.add ( new Block ( segment, TYPE_DECLARE )  );
                continue;
            }

            /* start a now block i.e. segment chain for the function body */
            final List<Segment> block_segments = new ArrayList<Segment>();
            block_segments.add ( segment );

            /* braces are guaranteed to be the last characters in Segments
             * if there are any */
            if ( ( str.contains("{") && last_char != '{' ) ||
                 ( str.contains("}") && last_char != '}' ) ||
                 // or if there are more than one curly brace of either kind
                 str.length() - str.replace("{", "").length() > 1 ||
                 str.length() - str.replace("}", "").length() > 1
            )
            {
                throw new RuntimeException( "[BlockParser.java] It seems like SegmentPaser doesn't work correctly. A given segment contains curly braces, but should be split at those: \n" + str );
            }

            /* loop and add segments until end of function body reached */
            int brace_count = 0;
            /* if last char is not a { it could also be a declaration spanning
             * multiple lines, e.g.
             *
             *     void myLongFunction(
             *         int const firstParameter,
             *         int const secondParameter
             *     );
             */
            if ( last_char == '{' )
                brace_count++;
            for ( int iInnerSegment = i + 1; iInnerSegment < segments.size(); ++iInnerSegment )
            {
                final Segment curSeg = segments.get ( iInnerSegment );
                /* clean code of comments (I don't think this is necessary),
                 * but it might be that uncommented function calls could be
                 * parsed as false positives for calls i.e. thinking the
                 * function is necessary where it isn't */
                if ( curSeg.getType() == SegmentParser.TYPE_COMMENT )
                    continue;

                block_segments.add ( curSeg );
                final String str2 = curSeg.getString().trim();
                if ( str2.isEmpty()  )
                    continue;
                final char last_char2 = str2.charAt ( str2.length() - 1 );

                if ( last_char2 == ';' && brace_count == 0 )
                {
                    ret.add ( new Block( block_segments, TYPE_DECLARE )  );
                    i = iInnerSegment;
                    break;
                }

                if ( last_char2 == '{' )
                    brace_count++;
                else if ( last_char2 == '}' )
                {
                    brace_count--;
                    if ( brace_count == 0 )
                    {
                        ret.add ( new Block( block_segments, TYPE_METHOD )  );
                        /* skip those segments we already added in this inner loop */
                        i = iInnerSegment;
                        break;
                    }
                }
            }
        }
        return ret;
    }
}
