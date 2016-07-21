/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.entry;


import java.util.Iterator;
import soot.rbclassload.HierarchySootClass;
import soot.rbclassload.HierarchySootMethod;
import soot.rbclassload.MethodTester;

/**
 * Simple functional which tests if a given method contains void gpuMethod
 * and implements the Kernel interface i.e. if it needs to be parsed by
 * Rootbeer.
 */
public class KernelEntryPointDetector implements MethodTester
{
    private boolean m_runTests;

    public KernelEntryPointDetector( boolean run_tests ) { m_runTests = run_tests; }

    public boolean test( HierarchySootMethod sm )
    {
        /* getSubSignature: Returns the Soot subsignature of this method. Used to refer to methods unambiguously. */
        if ( ! sm.getSubSignature().equals( "void gpuMethod()" ) )
            return false;

        final HierarchySootClass soot_class = sm.getHierarchySootClass();
        if ( ! m_runTests )
        {
            if ( soot_class.getName().startsWith("org.trifort.rootbeer.testcases.") )
                return false;
        }

        Iterator<String> iter = soot_class.getInterfaces().iterator();
        while ( iter.hasNext() )
        {
            String iface = iter.next();
            if ( iface.equals("org.trifort.rootbeer.runtime.Kernel") )
                return true;
        }
        return false;
    }
}
