/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.deadmethods;


import java.util.ArrayList;
import java.util.List;


public class MethodAnnotator
{
    public void parse
    (
        final List<Block > blocks,
        final List<String> method_names
    )
    {
        for ( final Block block : blocks )
        {
            if ( ! block.isMethod() )
                continue;

            /* buffer for performance */
            final String methodName    = block.getMethod().getName();
            final String str           = block.getFullStringNoStrings();
            final List<String> invoked = new ArrayList<String>();

            /* try to find any of the given method names in the method body,
             * i.e. they are invoked */
            for ( final String method_name : method_names )
            {
                /* ignore recursive calls (and also the function signature) */
                if ( method_name.equals( methodName ) )
                    continue;

                int start_pos = 0;
                while ( true ) /* why is pos == 0 excluded? I.e. the function call is at the beginning of the line? */
                {
                    int matchPos = str.indexOf( method_name, start_pos );
                    if ( matchPos < 1 )
                        break;

                    /* the found method name is only part of a larger name,
                     * so ignore it and try to find the next match if there are
                     * any
                     * shouldn't this also check for underscore? Or might this
                     * break something? */
                    final char c1 = str.charAt( matchPos - 1 );
                    if ( Character.isLetter(c1) || Character.isDigit(c1) || c1 == '_' )
                    {
                        start_pos += method_name.length();
                        continue;
                    }

                    /* check if the found match is possibly only a prefix of
                     * some longer method name or something different, but
                     * not a call. I.e. search for '(' */
                    int postMatchPos = matchPos + method_name.length();
                    boolean foundCall = false;
                    while ( postMatchPos < str.length() )
                    {
                        final char c2 = str.charAt( postMatchPos );
                        if ( c2 == ' ' || c2 == '\t' || c2 == '\n' )
                        {
                            postMatchPos++;
                            continue;
                        }
                        else if ( c2 == '(' )
                        {
                            foundCall = true;
                            invoked.add( method_name );
                            break;
                        }
                        else
                        {
                            /* we can skip the whole found method name match,
                             * because if we would skip less the smallest next
                             * match will surely be only a suffix of some larger
                             * function name */
                            start_pos += method_name.length();
                            break;
                        }
                    }
                    if ( foundCall )
                        break;
                }
            }
            block.getMethod().setInvoked( invoked );
        }
    }
}
