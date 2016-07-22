/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.compiler;


import java.util.List;

import org.trifort.rootbeer.generate.bytecode.GenerateForKernel;
import org.trifort.rootbeer.generate.opencl.OpenCLScene;

import soot.*;
import soot.rbclassload.DfsInfo;
import soot.rbclassload.RootbeerClassLoader;


public class Transform2
{
    private int m_Uuid;

    public Transform2(){
        m_Uuid = 1;
    }

    public void run( final String cls )
    {
        final OpenCLScene scene = new OpenCLScene();
        OpenCLScene.setInstance( scene );
        scene.init();

        final SootMethod method = Scene.v().getSootClass( cls ).getMethod( "void gpuMethod()" );

        final String uuid = getUuid();
        GenerateForKernel generator = new GenerateForKernel( method, uuid );
        try {
            generator.makeClass();
        } catch ( Exception ex )
        {
            ex.printStackTrace();
            OpenCLScene.releaseV();
            return;
        }

        //add an interface to the class
        SootClass soot_class = method.getDeclaringClass();
        SootClass iface_class = Scene.v().getSootClass("org.trifort.rootbeer.runtime.CompiledKernel");
        soot_class.addInterface(iface_class);

        System.out.println("added interface CompiledKernel");

        OpenCLScene.releaseV();
    }

    private String getUuid()
    {
        int uuid = m_Uuid;
        m_Uuid++;
        return Integer.toString(uuid);
    }
}
