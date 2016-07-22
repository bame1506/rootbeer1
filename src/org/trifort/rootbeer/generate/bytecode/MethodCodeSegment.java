/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.bytecode;


import java.util.ArrayList;
import java.util.List;

import org.trifort.rootbeer.generate.opencl.OpenCLMethod;
import org.trifort.rootbeer.generate.opencl.OpenCLScene;

import soot.Local;
import soot.SootClass;
import soot.SootMethod;
import soot.Type;
import soot.Value;
import soot.jimple.internal.JimpleLocal;


public class MethodCodeSegment
{
    private SootMethod m_existingMethod;
    private SootMethod m_clonedMethod  ;
    private SootClass  m_clonedClass   ;

    public MethodCodeSegment( SootMethod method ){ m_existingMethod = method; }

    public List<Local> getInputArguments()
    {
        final Type t = m_existingMethod.getDeclaringClass().getType();
        /* JimpleLocal creates a new variable of name 'r0'. You can see this
         * in generated_unix.cu. It seems like it is used for saving the this
         * pointer, that's why the type for r0 is that of the parent class.
         * I.e. the parent of gpuMethod i.e. the class which implements the
         * Kernel interface */
        List<Local> ret = new ArrayList<Local>();
        ret.add( new JimpleLocal( "r0", t ) );
        return ret;
    }

    public List<Local> getOutputArguments () { return new ArrayList<Local>()              ; }
    public SootClass   getSootClass       () { return m_existingMethod.getDeclaringClass(); }
    public List<Value> getInputValues     () { return new ArrayList<Value>()              ; }
    public SootMethod  getRootMethod      () { return m_existingMethod                    ; }
    public SootClass   getRootSootClass   () { return m_existingMethod.getDeclaringClass(); }

    /**
     * Returns the class where gpuMethod resides in as a Type.
     * Not sure why this and the other methods in this class return lists,
     * when the lists are always only one element long
     */
    public List<Type> getParameterTypes()
    {
        final List<Type> ret = new ArrayList<Type>();
        ret.add( m_existingMethod.getDeclaringClass().getType() );
        return ret;
    }

    public void makeCpuBody(SootClass soot_class) { m_clonedClass = soot_class; }
}
