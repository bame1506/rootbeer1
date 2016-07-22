/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.entry;

import soot.rbclassload.HierarchySootMethod;
import soot.rbclassload.MethodTester;

/**
 * Returns a positive for main( String[] ) methods
 */
public class MainTester implements MethodTester
{
    public boolean test( HierarchySootMethod hsm )
    {
        if ( hsm.getSubSignature().equals("void main(java.lang.String[])") )
            return true;
        else
            return false;
    }
}
