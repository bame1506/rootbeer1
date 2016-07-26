/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.deadmethods;


import java.util.ArrayList;
import java.util.List;


/**
 * Manages functional units of Segments, i.e. e.g. multiline declarations
 * or function definitions.
 */
public class Block
{
    private final List<Segment> m_segments           ;
    private final int           m_type               ;
    /* struct storing the extracted name of this method and a list of methods
     * this method uses. Set by MethodNameParser and MethodAnnotator */
    private       Method        m_method             ;
    /* Contains the whole function definition code. Partially without newlines */
    private       String        m_fullString         ;
    /* omits C strings i.e. "thisIsAString". Not sure what this is needed for.
     * The resulting code should not compile */
    private       String        m_fullStringNoStrings;

    public Block( final List<Segment> segments, final int type )
    {
        m_segments = segments;
        m_type     = type;

        if ( type == BlockParser.TYPE_METHOD )
        {
            final StringBuilder builder1 = new StringBuilder();
            final StringBuilder builder2 = new StringBuilder();

            for ( int i = 0; i < segments.size(); ++i )
            {
                final Segment segment = segments.get(i);
                builder1.append( segment.getString() );
                builder1.append( " " );
                if ( segment.getType() != SegmentParser.TYPE_STRING )
                {
                    builder2.append( segment.getString() );
                    builder2.append( " " );
                }
            }
            m_fullString          = builder1.toString();
            m_fullStringNoStrings = builder2.toString();
        }
    }

    /* Blocks with only a single segment can't be method definitions, because
     * segments are split at { and } resulting in at least two segments.
     * This constructor is used isnteda for top-level preprocessor lines */
    public Block( final Segment segment, final int type )
    {
        m_segments            = new ArrayList<Segment>();
        m_type                = type;
        m_segments.add( segment );
    }

    public int           getType      (){ return m_type      ; }
    public List<Segment> getSegments  (){ return m_segments  ; }
    public String        getFullString(){ return m_fullString; }
    public Method        getMethod    (){ return m_method    ; }
    public boolean       isMethod     (){ return getType() == BlockParser.TYPE_METHOD; }
    public String getFullStringNoStrings(){ return m_fullStringNoStrings; }
    public void setMethod(Method method){ m_method = method; }

    /* This is only needed for the debug output in DeadMethods.java, nothing
     * else!
     * And also possibly for building the final output of live methods / functions
     */
    @Override
    public String toString()
    {
        /* append a newline to preprocessor lines, because it was stripped off
         * while parsing */
        StringBuilder ret = new StringBuilder();
        if ( m_type == BlockParser.TYPE_DECLARE ||
             m_type == BlockParser.TYPE_DEFINE   )
        {
            for ( int i = 0; i < m_segments.size(); ++i )
            {
                Segment segment = m_segments.get(i);
                ret.append( segment.getString() );
                ret.append( "\n" );
            }
        }
        else
        {
            for ( int i = 0; i < m_segments.size(); ++i )
            {
                final Segment segment = m_segments.get(i);
                Segment nsegment = i < m_segments.size() - 1 ? m_segments.get(i+1) : null;
                ret.append( segment.getString() );
                /* only add a newline if this and the following segment are not
                 * strings @todo why not? */
                if ( segment.getType() != SegmentParser.TYPE_STRING &&
                     ( nsegment != null &&
                       nsegment.getType() != SegmentParser.TYPE_STRING
                     ) )
                {
                    ret.append( "\n" );
                }
            }
        }
        return ret.toString();
    }
}
