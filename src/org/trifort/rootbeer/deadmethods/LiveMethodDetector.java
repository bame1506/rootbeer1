/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.deadmethods;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;


/**
 * Inverses a graph like name -> (calls) -> List[Names] to
 * name <- (is called by) <- List[Names] or rather return a Set of
 * actually called methods. I.e. to breadth first search starting at entry
 * following all method invocations.
 */
public class LiveMethodDetector
{
    /**
     * @param[in] blocks a list of blocks whose method member is set with
     *                   a list of invoked names. I.e. need to run
     *                   MethodAnnotator beforehand.
     * @return set of method names actually visited starting at entry and run
     *         method
     */
    public static Set<String> parse( final List<Block> blocks )
    {
        final Map<String, Method> method_map = new HashMap<String, Method>();
        for ( final Block block : blocks )
        {
            if ( ! block.isMethod() )
                continue;
            final Method method = block.getMethod();
            method_map.put( method.getName(), method );
        }

        /* add run and entry to queue. Work through queue while queueing
         * all invoked function names. Repeating until all visited. Only
         * visiting unvisited in order not to hang cyclic calls */
        final LinkedList<String> queue = new LinkedList<String>();
        Set<String> visited = new HashSet<String>();
        queue.add("entry");
        queue.add("run");
        while ( ! queue.isEmpty() )
        {
            final String name = queue.removeFirst();
            if ( visited.contains( name ) )
                continue;
            visited.add( name );
            final Method method = method_map.get( name );
            if ( method == null )
                continue;
            queue.addAll( method.getInvoked() );
        }
        return visited;
    }
}
