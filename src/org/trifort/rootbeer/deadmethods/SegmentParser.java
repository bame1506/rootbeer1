/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.deadmethods;


import java.util.ArrayList;
import java.util.List;


public class SegmentParser
{
    public static final int TYPE_FREE    = 0;
    public static final int TYPE_COMMENT = 1;
    public static final int TYPE_STRING  = 2;
    public static final int TYPE_CHAR    = 3;
    public static final int TYPE_DEFINE  = 4;

    /* this is only used for the internal parser and will be converted to
     * TYPE_COMMENT */
    private static final int TYPE_MULTLINE_COMMENT = 5;

    public SegmentParser(){}

    /**
     * parses a given string of C code into functional segments i.e. partial
     * strings with an ID specifying whether they are comments or strings or
     * plain code.
     *
     * Example input:

     *    // Convert integer part
     *    do {
     *      convert[place++] = "0123456789abcdef"[uvalue % (unsigned)base];
     *      uvalue = (uvalue / (unsigned)base );
     *    }
     *
     * Example Output:
     *
     *    TYPE_COMMENT: // Convert integer part
     *    TYPE_FREE:   do {
     *    TYPE_FREE:     convert[place++] =
     *    TYPE_STRING: "0123456789abcdef"
     *    TYPE_FREE: [uvalue % (unsigned)base];
     *    TYPE_FREE:     uvalue = (uvalue / (unsigned)base );
     *    TYPE_FREE:   }
     */
    public List<Segment> parse( final String contents )
    {
        final List<Segment> ret = new ArrayList<Segment>();
        int state = TYPE_FREE;
        StringBuilder accum = new StringBuilder();

        /* parse each character in given string */
        for ( int i = 0; i < contents.length(); ++i )
        {
            final char c  = contents.charAt(i);          /* current character */
            final char cc = i < contents.length() - 1 ?  /* next character    */
                      contents.charAt( i+1 ) : '\0';

            /* @todo: Redo this with regex shorter?
             *        ( (//[^\n]*|"[^"]*"|'[^']*'|\n[ \t]*#[^\n]*|.*[{}])
             *                      ^ problem can't specify escaped " \" or \\\"
             */
            switch ( state )
            {
                case TYPE_FREE:
                {
                    /* If we found either of these, then write out
                     * everything in accum until now with the current state
                     * which is TYPE_FREE. And begin a new accum which
                     * already contains the matched characters */
                    /* Found doubleslash for comment // */
                    if ( c == '/' && cc == '/' )
                    {
                        if ( accum.length() > 0 )
                            ret.add( new Segment( accum.toString(), state ) );
                        accum = new StringBuilder("//");
                        state = TYPE_COMMENT;
                        ++i; // ignore next character (slash)
                    }
                    else if ( c=='/' && cc == '*' )
                    {
                        if ( accum.length() > 0 )
                            ret.add( new Segment( accum.toString(), state ) );
                        accum = new StringBuilder("/*");
                        state = TYPE_MULTLINE_COMMENT;
                        ++i; // ignore next character (asterisk)
                    }
                    else if ( c == '\"' )
                    {
                        if ( accum.length() > 0 )
                            ret.add( new Segment( accum.toString(), state ) );
                        accum = new StringBuilder( ""+c );
                        state = TYPE_STRING;
                    }
                    else if ( c == '\'' )
                    {
                        if ( accum.length() > 0 )
                            ret.add( new Segment( accum.toString(), state ) );
                        accum = new StringBuilder( ""+c );
                        state = TYPE_CHAR;
                    }
                    /**
                     * Problem: accum can be empty after some other segment
                     *          finished. i.e. printf( "test" #3 );
                     *          or do { } #pragma
                     *          But I guess this isn't correct C in the first
                     *          place
                     */
                    else if ( c == '#' && onlyWhitespace(accum) )
                    {
                        if ( accum.length() > 0 )
                            ret.add( new Segment( accum.toString(), state ) );
                        accum = new StringBuilder( ""+c );
                        state = TYPE_DEFINE;
                    }
                    /* opening and closing brackets each belong to the last
                     * code segment.
                     * Splitting at {} is necessary in order to correctly
                     * recognize code blocks @see BlockParser.java
                     */
                    else if ( c == '}' || c == '{' )
                    {
                        accum.append(c);
                        ret.add( new Segment( accum.toString(), state ) );
                        accum = new StringBuilder("");
                    }
                    /* on newline break code segments, start a new one, this
                     * is necessary to recognize # which may only be at the
                     * beginning of a line */
                    else if ( c == '\n' )
                    {
                        if ( accum.length() > 0 )
                            ret.add( new Segment( accum.toString(), state ) );
                        accum = new StringBuilder("");
                    }
                    else
                        accum.append(c);
                    break;
                } // case TYPE_FREE
                case TYPE_MULTLINE_COMMENT:
                {
                    /* only a * / may end a multline comment. Note that *\'\n'/
                     * also counts as a comment ended, but this exotic case
                     * shall be ignored */
                    if ( c == '*' && cc == '/' )
                    {
                        accum.append( "*/" );
                        ++i; // ignore next char (cc)
                        ret.add( new Segment( accum.toString(), TYPE_COMMENT ) );
                        accum = new StringBuilder();
                        state = TYPE_FREE;
                    }
                    else
                        accum.append(c);
                    break;
                }
                case TYPE_COMMENT:
                {
                    /* only a line break may end an end-of-line comment, except
                     * if the linebreak is escaped */
                    if ( c == '\n' )
                    {
                        if ( insideEscape(contents, i - 1) )
                            accum.append(c);
                        else
                        {
                            if ( accum.length() > 0 )
                                ret.add( new Segment( accum.toString(), state ) );
                            accum = new StringBuilder();
                            state = TYPE_FREE;
                        }
                    }
                    else
                        accum.append(c);
                    break;
                }
                case TYPE_STRING:
                case TYPE_CHAR:
                {
                    accum.append(c);
                    if ( ( ( state == TYPE_STRING && c == '\"' ) ||
                           ( state == TYPE_CHAR   && c == '\'' )  ) &&
                         ! insideEscape( contents, i - 1 ) )
                    {
                        ret.add( new Segment( accum.toString(), state ) );
                        accum = new StringBuilder();
                        state = TYPE_FREE;
                    }
                    break;
                }
                case TYPE_DEFINE:
                {
                    /* nothing except newline may end a define segment
                     * @todo But what is with comments after or even before
                     *       a define ??? In the end this parser is only used
                     *       on the template c-files anyway, so no "faulty" user
                     *       must be anticipated
                     */
                    if ( c == '\n' )
                    {
                        /**
                         * test if the newline is escaped e.g. like normally
                         * done for longer #defines. An example where this may
                         * be important:
                         *   #define TEST(x) \
                         *   printf( "C:\\Windows\\System32\\\
                         *   print.dll" );
                         * Although this is bad style anyway.
                         */
                        if ( insideEscape( contents, i - 1 ) )
                            accum.append(c);
                        else
                        {
                            /* don't append newline */
                            if ( accum.length() > 0 )
                                ret.add( new Segment( accum.toString(), state ) );
                            accum = new StringBuilder();
                            state = TYPE_FREE;
                        }
                    } else
                        accum.append(c);
                    break;
                }
                default:
                    throw new RuntimeException( "[SegmentParser.java] variable state assumed an illegal state!" );
            }
        } // for c : string

        /* add the last segment */
        if ( accum.length() > 0 )
            ret.add( new Segment( accum.toString(), state ) );

        return ret;
    }

    /**
     * Count the consecutive backslashes before the given index and
     * return true if they are not a multiple of two
     *
     * e.g. \\'\n' returns true for index 2, because it means '\n' is not
     * escaped
     */
    private static boolean insideEscape( final String contents, final int index )
    {
        int count = 0;
        for ( int i = index; i >= 0; --i )
        {
            if ( contents.charAt(i) == '\\' )
                ++count;
            else
                break;
        }
        return count % 2 != 0 ? true : false;
    }

    /**
     * Tests if only contains whitespaces.
     * Necessary because whitespaces prior to # do not matter, but other
     * characters do
     */
    private static boolean onlyWhitespace( final StringBuilder builder )
    {
        for ( int i = 0; i < builder.length(); ++i )
        {
            final char c = builder.charAt(i);
            if ( ! ( c == ' ' || c == '\n' || c == '\t' || c == '\r' ) )
                return false;
        }
        return true;
    }
}
