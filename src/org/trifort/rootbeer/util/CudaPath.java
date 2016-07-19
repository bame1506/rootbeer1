/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A class which tries all sorts of trickery and sorcery to locate the CUDA
 * binary directory (with trailing /).
 */
public class CudaPath
{
    private static String[] m_windowsSearchPaths = {
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\",
        "C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA\\"
    };
    private static String[] m_unixSearchPaths = {
        "/usr/local/cuda/bin/",
        "/usr/local/cuda-5.5/bin/",
        "/usr/lib/nvidia-cuda-toolkit/bin/"
    };

    public static String get() {
        if ( File.separator.equals("/") )
          return getUnix();
        else
          return getWindows();
    }

    /***** Private Methods *****/

    private static boolean isValidBinary( String path )
    {
        File file = new File( path );
        return file.exists() && file.isFile() && file.canRead();
    }

    private static String getUnix()
    {
        /* Search CUDA_BIN_PATH environment variable (seems to be a pre-5.5 thing) */
        if ( System.getenv().containsKey("CUDA_BIN_PATH") )
        {
            String s = System.getenv("CUDA_BIN_PATH");
            if ( ! s.endsWith("/") )
                return s + "/";
            return s;
        }

        /* Search path variable using 'which' command.
         * @todo: Could be tried manually first in case which is not available */
        BufferedReader input = null; // stores output of 'which nvcc'
        try
        {
            input = new BufferedReader( new InputStreamReader(
                Runtime.getRuntime().exec("which nvcc").getInputStream() ) );
            String output = input.readLine();
            if ( output != null && ! output.isEmpty() )
            {
                output = output.trim(); // leading and trailing whitespaces
                if ( isValidBinary( output ) )
                {
                    // strip 'nvcc' exe-name, we only want the path
                    return output.substring( 0, output.lastIndexOf("nvcc") );
                }
            }
        }
        catch ( IOException e ) {} // File not found => do nothing, go to next part
        finally
        {
            try {
                input.close();
            /* If this fails there is nothing we can do, but it is not
             * important for Rootbeer, so keep running */
            } catch ( Exception e ) {}
        }

        /* Search path using 'wheris' command:
         *   "Locate the binary, source, and manual-page files for a command."
         * Output of 'whereis nvcc' could be:
         *   nvcc: /opt/cuda-7.0/bin/nvcc /opt/cuda-7.0/bin/nvcc.profile /usr/share/man/man1/nvcc.1
         */
        input = null;
        try
        {
            input = new BufferedReader( new InputStreamReader(
                Runtime.getRuntime().exec("whereis nvcc").getInputStream() ) );
            String output = input.readLine();
            if ( output != null )
            {
                /* splitting at space may lead to bugs with paths containing
                 * spaces, but that is a problem with whereis in the first
                 * place */
                String[] sp = output.split(" ");
                for ( String s: sp )
                {
                    s = s.trim();
                    if ( s.endsWith("nvcc") && isValidBinary( s+"nvcc" ) )
                        return s.substring( 0, s.lastIndexOf("nvcc") );
                }
            }
        } catch ( IOException e )
            {} // Do nothing, go to next part. This can be thrown by exec
        finally
        {
            try {
                input.close();
            } catch ( Exception e ) {} // If this fails there is nothing we can do
        }

        /* Search some standard paths */
        for ( String path : m_unixSearchPaths )
        {
            if ( isValidBinary( path+"nvcc" ) )
              return path;
        }

        throw new RuntimeException( "Could not find nvcc binary, please check that you have CUDA installed in such a fasion that 'which nvcc' can find it!" );
    }

    private static String getWindows()
    {
        for ( String path : m_windowsSearchPaths )
        {
            String nvcc = findFileRecursively( path, "nvcc.exe" );
            if ( nvcc != null )
                return nvcc;
        }
        if ( System.getenv().containsKey("CUDA_BIN_PATH") )
            return findFileRecursively( System.getenv("CUDA_BIN_PATH"), "nvcc.exe" );

        throw new RuntimeException( "cannot find nvcc.exe. Try setting the CUDA_BIN_PATH to the folder with nvcc.exe" );
    }

    /**
     * Searches for a given file recursively in a directory.
     * Like unix command: find <directory> -name '<fileName>'
     */
    private static String findFileRecursively
    (
        final String directory,
        final String fileName
    )
    {
        File file = new File( directory );
        if ( ! file.exists() )
            return null;
        if ( ! file.isDirectory() )
        {
            if ( file.getName().equals( fileName ) )
                return file.getAbsolutePath();
            else
                throw new RuntimeException( "[findFileRecursively] The first argument must be a path to a directory, not a file!" );
        }
        else
        {
            /* get all files in directory and cycle through them */
            File[] children = file.listFiles();

            /* sort file names (using String.compareTo) in order to get
             * some kind of determined behavior if there are multiple files
             * with the same name in different subdirectories */
            FileSorter[] sorted_children = new FileSorter[children.length];
            for ( int i = 0; i < children.length; ++i )
                sorted_children[i] = new FileSorter( children[i] );
            Arrays.sort( sorted_children );
            for ( int i = 0; i < sorted_children.length; ++i )
                children[i] = sorted_children[i].getFile();

            for ( File child : children )
            {
                if ( child.isDirectory() )
                {
                    /* Call recursively and if found, return that result */
                    String nvcc = findFileRecursively( child.getAbsolutePath(), fileName );
                    if ( nvcc != null )
                        return nvcc;
                } else if ( child.getName().equals( fileName ) && child.canRead() )
                    return child.getAbsolutePath();
            }
        }

        return null;
    }

    /* Boiler plate to make File comparable using String comparators,
     * so that sort can be used */
    private static class FileSorter implements Comparable<FileSorter>
    {
        private File m_file;

        public FileSorter( File file ) { m_file = file; }
        public int compareTo( FileSorter o )
        {
            String lhs = m_file.getAbsolutePath();
            String rhs = o.m_file.getAbsolutePath();

            return rhs.compareTo( lhs );
        }
        public File getFile() { return m_file; }
    }
}
