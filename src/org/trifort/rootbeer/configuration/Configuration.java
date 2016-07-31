/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.configuration;


import java.util.List;

import org.trifort.rootbeer.generate.opencl.tweaks.GencodeOptions.CompileArchitecture;
import org.trifort.rootbeer.generate.opencl.tweaks.GencodeOptions.ComputeCapability;
import org.trifort.rootbeer.util.ResourceReader;


/**
 * Basically a C++ struct + boilerplate which saves the configuration
 * of a kernel(?) like what CUDA architecture it depends on or the register
 * count used.
 */
public class Configuration
{
    public static final int MODE_GPU  = 0;
    public static final int MODE_NEMU = 1;
    public static final int MODE_JEMU = 2;

    /* singleton structure */
    private static Configuration m_Instance;
    public static Configuration compilerInstance()
    {
        if ( m_Instance == null )
            m_Instance = new Configuration();
        return m_Instance;
    }

    /**
     * Only used by CUDAContext.java. Like compilerInstance, but creates a
     * new singleton if configuration was loaded from config.txt
     */
    public static Configuration runtimeInstance()
    {
        if ( m_Instance == null )
            m_Instance = new Configuration(true);
        else if ( m_Instance.m_compilerInstance )
            m_Instance = new Configuration(true);
        return m_Instance;
    }

    private static boolean      m_runAll            ;
    private static boolean      m_printMem          ;
    private int                 m_mode              ;
    private boolean             m_compilerInstance  ;
    private boolean             m_remapAll          ;
    private boolean             m_maxRegCountSet    ;
    private int                 m_maxRegCount       ;
    private boolean             m_arrayChecks       ;
    private boolean             m_doubles           ;
    private boolean             m_recursion         ;
    private boolean             m_exceptions        ;
    private boolean             m_keepMains         ;
    private int                 m_sharedMemSize     ;
    private CompileArchitecture m_arch              ;
    private boolean             m_manualCuda        ;
    private String              m_manualCudaFilename;
    private ComputeCapability   m_computeCapability ;

    static {
        m_printMem = false;
    }

    private Configuration()
    {
        m_compilerInstance  = true;
        m_remapAll          = true;
        m_maxRegCountSet    = false;
        m_arrayChecks       = true;
        m_doubles           = true;
        m_recursion         = true;
        m_exceptions        = true;
        m_keepMains         = false;
        m_sharedMemSize     = 40*1024;
        m_arch              = CompileArchitecture.Arch32bit64bit;
        m_manualCuda        = false;
        m_computeCapability = ComputeCapability.ALL;
    }

    private Configuration( boolean load )
    {
        m_compilerInstance = false;
        try {
            List<byte[]> data = ResourceReader.getResourceArray( "/org/trifort/rootbeer/runtime/config.txt" );
            m_mode       = data.get(0)[0];
            m_exceptions = data.get(0)[1] == 1 ? true : false;
        } catch( Exception ex ) {
            m_mode = MODE_GPU;
        }
    }

    public static void setRunAllTests  ( boolean run_all ) { m_runAll         = run_all; }
    public static void setPrintMem     ( boolean print   ) { m_printMem       = print  ; }
    public void        setMode         ( int mode        ) { m_mode           = mode   ; }
    public void        setRemapSparse  (                 ) { m_remapAll       = false  ; }
    public void        setMaxRegCount  ( int value       ) { m_maxRegCount    = value  ;
                                                             m_maxRegCountSet = true   ; }
    public void        setArrayChecks  ( boolean value   ) { m_arrayChecks    = value  ; }
    public void        setDoubles      ( boolean value   ) { m_doubles        = value  ; }
    public void        setRecursion    ( boolean value   ) { m_recursion      = value  ; }
    public void        setExceptions   ( boolean value   ) { m_exceptions     = value  ; }
    public void        setKeepMains    ( boolean value   ) { m_keepMains      = value  ; }
    public void        setSharedMemSize( int size        ) { m_sharedMemSize  = size   ; }
    public void        setManualCuda   (                 ) { m_manualCuda     = true   ; }
    public void    setCompileArchitecture( CompileArchitecture arch ) { m_arch = arch; }
    public void    setManualCudaFilename ( String filename) { m_manualCudaFilename = filename; }
    public void    setComputeCapability  ( ComputeCapability computeCapability) { m_computeCapability = computeCapability; }

    public boolean             isMaxRegCountSet      () { return m_maxRegCountSet    ; }
    public boolean             isManualCuda          () { return m_manualCuda        ; }
    public int                 getMode               () { return m_mode              ; }
    public boolean             getRemapAll           () { return m_remapAll          ; }
    public static boolean      getRunAllTests        () { return m_runAll            ; }
    public static boolean      getPrintMem           () { return m_printMem          ; }
    public int                 getMaxRegCount        () { return m_maxRegCount       ; }
    public boolean             getArrayChecks        () { return m_arrayChecks       ; }
    public boolean             getDoubles            () { return m_doubles           ; }
    public boolean             getRecursion          () { return m_recursion         ; }
    public boolean             getExceptions         () { return m_exceptions        ; }
    public boolean             getKeepMains          () { return m_keepMains         ; }
    public int                 getSharedMemSize      () { return m_sharedMemSize     ; }
    public CompileArchitecture getCompileArchitecture() { return m_arch              ; }
    public String              getManualCudaFilename () { return m_manualCudaFilename; }
    public ComputeCapability   getComputeCapability  () { return m_computeCapability ; }
}
