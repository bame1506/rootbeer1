/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.deadmethods;


import java.util.ArrayList;
import java.util.List;


public class MethodNameParser
{
    /**
     * returns a list of strings containing the method names to the list
     * of blocks being given. Note that non-method definition blocks are
     * ignored. I.e. the returned list might contain fewer strings than blocks
     */
    public static List<String> parse( final List<Block> blocks )
    {
        final List<String> ret = new ArrayList<String>();
        for ( Block block : blocks )
        {
            if ( block.isMethod() )
            {
                final String name = parseMethodName( block.getFullString() );
                ret.add(name);
                block.setMethod( new Method(name) );
            }
        }
        return ret;
    }

    /**
     * Equivalent to this sed regex command: s|([^ ]*\(|\1|p
     * I.e. find the first word in front of the first opening parenthesis (
     * @todo use java regex
     */
    private static String parseMethodName( final String str )
    {
        int first_char_pos = str.indexOf('(') - 1;
        while ( first_char_pos >= 0 )
        {
            if ( str.charAt( first_char_pos ) != ' ' )
                break;
            --first_char_pos;
        }

        int first_space_pos = first_char_pos - 1;
        while ( first_space_pos >= 0 )
        {
            if ( str.charAt( first_space_pos ) == ' ' )
                break;
            --first_space_pos;
        }

        return str.substring( first_space_pos+1, first_char_pos+1 );
    }
}
