/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.opencl.tweaks;

import java.io.File;
import java.util.List;

import org.trifort.rootbeer.configuration.Configuration;
import org.trifort.rootbeer.util.CmdRunner;
import org.trifort.rootbeer.util.CudaPath;

public class GencodeOptions
{
    public enum CompileArchitecture {
        Arch32bit, Arch64bit, Arch32bit64bit;
    }

    public enum ComputeCapability {
        ALL, SM_11, SM_12, SM_20, SM_21, SM_30, SM_35, SM_50, SM_52, SM_60, SM_61;
    }

    private boolean versionMatches( String versionString, String version ) {
        return versionString.contains( "release "+version );
    }

    /**
     * Returns one large string of options asking nvcc to compile for all
     * known architectures at the given nvcc version.
     *
     * @see http://docs.nvidia.com/cuda/maxwell-compatibility-guide/#building-maxwell-compatible-apps-using-cuda-5-5
     */
    public String getOptions()
    {
        /* Version will be something like 75 for 7.5 or 50 for 5.0 */
        final int version = getNVCCVersion();

        String sm_35;
        String sm_30;
        String sm_21;
        String sm_20;
        String sm_12;
        String sm_11;
        if ( File.separator.equals("/") ) /* if not Windows */
        {
            sm_35 = "--generate-code arch=compute_35,code=\"sm_35,compute_35\" ";
            sm_30 = "--generate-code arch=compute_30,code=\"sm_30,compute_30\" ";
            sm_21 = "--generate-code arch=compute_20,code=\"sm_21,compute_20\" ";
            sm_20 = "--generate-code arch=compute_20,code=\"sm_20,compute_20\" ";
            sm_12 = "--generate-code arch=compute_12,code=\"sm_12,compute_12\" ";
            sm_11 = "--generate-code arch=compute_11,code=\"sm_11,compute_11\" ";
        } else {
            sm_35 = "--generate-code arch=compute_35,code=\"sm_35\" ";
            sm_30 = "--generate-code arch=compute_30,code=\"sm_30\" ";
            sm_21 = "--generate-code arch=compute_20,code=\"sm_21\" ";
            sm_20 = "--generate-code arch=compute_20,code=\"sm_20\" ";
            sm_12 = "--generate-code arch=compute_12,code=\"sm_12\" ";
            sm_11 = "--generate-code arch=compute_11,code=\"sm_11\" ";
        }

        //sm_12 doesn't support recursion
        if ( Configuration.compilerInstance().getRecursion() ) {
            sm_12 = "";
            sm_11 = "";
        }

        //sm_12 doesn't support doubles
        if ( Configuration.compilerInstance().getDoubles() ) {
            sm_12 = "";
            sm_11 = "";
        }

        if ( 50 <= version && version <= 70 )
        {
            switch ( Configuration.compilerInstance().getComputeCapability() )
            {
                case ALL:   return sm_35 + sm_30 + sm_21 + sm_20 + sm_12 + sm_11;
                case SM_11: return sm_11;
                case SM_12: return sm_12;
                case SM_20: return sm_20;
                case SM_21: return sm_21;
                case SM_30: return sm_30;
                case SM_35: return sm_35;
                default:    return sm_35 + sm_30 + sm_21 + sm_20 + sm_12 + sm_11;
            }
        } else if ( 32 <= version && version <= 42 )
        {
            switch ( Configuration.compilerInstance().getComputeCapability() )
            {
                case ALL:   return sm_30 + sm_21 + sm_20 + sm_12 + sm_11;
                case SM_11: return sm_11;
                case SM_12: return sm_12;
                case SM_20: return sm_20;
                case SM_21: return sm_21;
                case SM_30: return sm_30;
                default:    return sm_30 + sm_21 + sm_20 + sm_12 + sm_11;
            }
        } else if ( version == 31 || version == 30 )
        {
            switch ( Configuration.compilerInstance().getComputeCapability() )
            {
                case ALL:   return sm_20 + sm_12 + sm_11;
                case SM_11: return sm_11;
                case SM_12: return sm_12;
                case SM_20: return sm_20;
                default:    return sm_20 + sm_12 + sm_11;
            }
        } else {
            throw new RuntimeException( "unsupported nvcc version. version 3.0 or higher needed. arch sm_11 or higher needed." );
        }
    }

    /**
     * Reads the nvcc version using nvcc --version
     *
     * Possible outputs are e.g.:
     *
     *   nvcc: NVIDIA (R) Cuda compiler driver
     *   Copyright (c) 2005-2012 NVIDIA Corporation
     *   Built on Fri_Sep_21_17:28:58_PDT_2012
     *   Cuda compilation tools, release 5.0, V0.2.1221
     *
     *   nvcc: NVIDIA (R) Cuda compiler driver
     *   Copyright (c) 2005-2014 NVIDIA Corporation
     *   Built on Thu_Jul_17_21:41:27_CDT_2014
     *   Cuda compilation tools, release 6.5, V6.5.12
     *
     *   nvcc: NVIDIA (R) Cuda compiler driver
     *   Copyright (c) 2005-2015 NVIDIA Corporation
     *   Built on Mon_Feb_16_22:59:02_CST_2015
     *   Cuda compilation tools, release 7.0, V7.0.27
     *
     *   nvcc: NVIDIA (R) Cuda compiler driver
     *   Copyright (c) 2005-2015 NVIDIA Corporation
     *   Built on Tue_Aug_11_14:27:32_CDT_2015
     *   Cuda compilation tools, release 7.5, V7.5.17
     */
    public static int getNVCCVersion()
    {
        CudaPath cuda_path = new CudaPath();
        String cmd[] = new String[2];
            cmd[0] = cuda_path.get();
            cmd[1] = "--version";
        if ( File.separator.equals("/") ) /* Linux, MacOS */
            cmd[0] += "nvcc";

        CmdRunner runner = new CmdRunner();
        runner.run( cmd, new File(".") );
        List<String> lines = runner.getOutput();
        if ( lines.isEmpty() )
        {
            List<String> error_lines = runner.getError();
            for ( String error_line : error_lines )
                System.err.println( error_line );
            throw new RuntimeException("Error detecting nvcc version.");
        }
        for ( int i = lines.size()-1; i > 0; --i )
        {
            if ( lines.get(i).matches( ".*release [0-9]\\.[0-9][^0-9].*" ) )
            {
                /* Note that split takes a regex. So we need to escape .!
                 * Or else the string will be split at every character,
                 * resulting in nothing remaining. */
                final String[] versionStrings = lines.get(i).
                    split( "release " )[1].substring( 0,3 ).split("\\.");
                return Integer.parseInt( versionStrings[0] ) * 10 +
                       Integer.parseInt( versionStrings[1] );
            }
        }
        throw new RuntimeException( "Error detecting nvcc version." );
    }
}
