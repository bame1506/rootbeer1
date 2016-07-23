/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.bytecode;


import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import org.trifort.rootbeer.compiler.FindMethodCalls;

import soot.Body;
import soot.SootClass;
import soot.SootField;
import soot.SootMethod;
import soot.Unit;
import soot.Value;
import soot.jimple.AssignStmt;
import soot.jimple.FieldRef;


/**
 * Caches the class names which are written to inside the code of the
 * gpuMethod given in the ReadOnlyTypes constructor
 */
public class ReadOnlyTypes
{
    private SootClass   m_RootClass     ;
    private Set<String> m_WrittenClasses;
    private Set<String> m_Inspected     ;

    public ReadOnlyTypes( final SootMethod gpuMethod )
    {
        m_RootClass      = gpuMethod.getDeclaringClass();
        m_WrittenClasses = new HashSet<String>();
        m_Inspected      = new HashSet<String>();
        inspectMethod(gpuMethod);
    }

    public boolean isRootReadOnly() { return isReadOnly( m_RootClass ); }

    public boolean isReadOnly( SootClass soot_class ) {
        return ! m_WrittenClasses.contains( soot_class.getName() );
    }

    /**
     * Just a wrapper for inspectBody which remembers which methods were
     * already analyzed and which checks if the methid has a valid body
     */
    private void inspectMethod( final SootMethod method )
    {
        /* I don't think this check should ever succeed, because
         * m_Inspected is set to an empty Set in the constructor right
         * before calling this method. Also this method is private */
        assert ! m_Inspected.contains( method.getSignature() );
        m_Inspected.add( method.getSignature() );

        if ( ! method.isConcrete() || ! method.hasActiveBody() )
            return;
        final Body body = method.getActiveBody();
        if ( body == null )
            return;
        inspectBody( body );

        final Iterator<SootMethod> iter = new FindMethodCalls().findForMethod( method ).iterator();
        while ( iter.hasNext() )
            inspectMethod( iter.next() );
    }

    /**
     * Iterates over each Soot statement in the method body, i.e. the user
     * code and searches for assignments to fields of classes and writes
     * the the classes which were assigned into m_WrittenClasses,
     * meaning they are not read-only.
     * This includes fields (i.e. member variables) of gpuMethod itself.
     * To check if these are written to, there exists 'isRootReadOnly'
     *
     * @see https://github.com/Sable/soot/wiki/Fundamental-Soot-objects
     */
    private void inspectBody( final Body body )
    {
        Iterator<Unit> iter = body.getUnits().iterator();
        while ( iter.hasNext() )
        {
            final Unit curr = iter.next();

            if ( ! ( curr instanceof AssignStmt ) )
                continue;

            final AssignStmt assign = (AssignStmt) curr;
            final Value lhs = assign.getLeftOp();

            if ( ! ( lhs instanceof FieldRef ) )
                continue;
            final FieldRef ref = (FieldRef) lhs;

            final String variableName = ref.getField().getDeclaringClass().getName();
            if ( ! m_WrittenClasses.contains( variableName ) )
                m_WrittenClasses.add( variableName );
        }
    }
}
