/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */


package org.trifort.rootbeer.deadmethods;


/**
 * Abstraction which save a string and the recognized type of what the string
 * represents in C source code. Could be a simple struct in C
 */
public class Segment
{
    private final String m_str ;
    private final int    m_type;

    public Segment( final String str, final int type )
    {
        m_str  = str;
        m_type = type;
    }

    public String getString(){ return m_str ; }
    public int    getType  (){ return m_type; }
}
