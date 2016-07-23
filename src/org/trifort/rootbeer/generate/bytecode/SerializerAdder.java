/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.bytecode;


import soot.*;


/**
 * Only used one damn time by GenerateForKernel.java
 */
public class SerializerAdder
{
    private String m_serializerClassName;

    public SerializerAdder(){}

    public void add( final MethodCodeSegment block )
    {
        System.out.println( "generating serialization bytecode..." );
        final VisitorGen generate_visitor = new VisitorGen( block.getRootSootClass() );
        generate_visitor.generate();
        m_serializerClassName = generate_visitor.getClassName();
    }

    public String getSerializerClassName() { return m_serializerClassName; }
}
