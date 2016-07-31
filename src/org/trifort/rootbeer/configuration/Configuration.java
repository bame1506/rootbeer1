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
 *
 * But weirdly it seems like it is never used at runtime, only for compiling
 * For runtime only getExceptions is used.
 */
public class Configuration
{
    public static final int MODE_GPU  = 0;
    public static final int MODE_NEMU = 1;
    public static final int MODE_JEMU = 2;

    /* singleton structure
     * Not compilerInstance is not thread-safe, but it's not that important
     * except for compiling many programs multithreaded, but then just use
     * multiple processes ...
     *
     * The uses in these files are a bitch to iron out:
     *   generate/bytecode/GenerateForKernel.java
     *   generate/opencl/OpenCLArrayType.java
     *   generate/opencl/OpenCLMethod.java
     *   generate/opencl/body/MethodStmtSwitch.java
     *   generate/opencl/body/OpenCLBody.java
     *   generate/opencl/fields/OpenCLField.java
     */
    private static Configuration m_Instance;
    public static void setInstance( final Configuration configuration ){ m_Instance = configuration; }
    public static Configuration compilerInstance()
    {
        assert( m_Instance != null );
        return m_Instance;
    }

    private static boolean      m_runAll            ;
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

    public Configuration()
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

    /**
     * Reads the compiler configuration from a config file packed in the
     * runtime jar. Currently the config only contains the mode which is
     * ignored on runtime and whether to use exceptions or not which is the
     * only 'bit' of configuration which is actually used.
     */
    public Configuration( boolean load )
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
