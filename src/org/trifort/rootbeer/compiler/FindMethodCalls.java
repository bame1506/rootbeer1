/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.compiler;


import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import soot.Body;
import soot.PatchingChain;
import soot.SootMethod;
import soot.Unit;
import soot.Value;
import soot.ValueBox;
import soot.jimple.InvokeExpr;


public class FindMethodCalls
{
    public FindMethodCalls(){}

    public static Set<SootMethod> findForBody( final Body body )
    {
        final Set<SootMethod> methods = new LinkedHashSet<SootMethod>();
        final Iterator<Unit> iter = body.getUnits().iterator();
        while ( iter.hasNext() )
        {
            Unit unit = iter.next();
            List<ValueBox> vboxes = unit.getUseAndDefBoxes();
            for ( ValueBox vbox : vboxes )
            {
                final Value value = vbox.getValue();
                if ( ! ( value instanceof InvokeExpr ) )
                    continue;
                final InvokeExpr expr = (InvokeExpr) value;
                final SootMethod method = expr.getMethod();
                if ( ! methods.contains( method ) )
                    methods.add( method );
            }
        }
        return methods;
    }

    public static Set<SootMethod> findForMethod( SootMethod method )
    {
        if ( ! method.isConcrete() )
            return new HashSet<SootMethod>();
        Body body = method.getActiveBody();
        return findForBody(body);
    }
}
