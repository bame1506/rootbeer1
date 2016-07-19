/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Executes a command a saves it's output.
 *
 * This class is necessary because of error handling and because of handling
 * that the process actually finishes. We don't want the output being streamed
 */
public class CmdRunner
{
    private List<String> m_outputLines;
    private List<String> m_errorLines ;
    private Process      m_process    ;

    public int run( String cmd, File dir )
    {
        /* call the run method with a String array. It's actually better to
         * never make use of this method, especially if the command contains
         * spaces in it! That is the same problem with exec. */
        return run( cmd.split(" "), dir );
    }

    public int run( String cmd[], File dir )
    {
        try
        {
            m_process = Runtime.getRuntime().exec( cmd, null, dir );
            return processExec();
        }
        catch ( Exception ex )
        {
            ex.printStackTrace();
            throw new RuntimeException(ex);
        }
    }

    private int processExec() throws InterruptedException
    {
        StreamEater out_eater = new StreamEater( m_process.getInputStream() );
        StreamEater err_eater = new StreamEater( m_process.getErrorStream() );
        m_outputLines = out_eater.get();
        m_errorLines  = err_eater.get();
        /* wait until the process finishes. actually it should already be
         * guaranteed that the process has finished after calling 'get',
         * because it waits for the program output stream to return null.
         * But with waitFor we can cat the exit code anyway.
         * Could use 'exitValue' here though. */
        int ret = m_process.waitFor();
        /* seems unnecessary to me, because waitFor should guarantee that
         * the process has already exited ... */
        m_process.destroy();
        return ret;
    }

    public List<String> getOutput() { return m_outputLines; }
    public List<String> getError () { return m_errorLines ; }

    /**
     * Similar to Java's BufferedReader, but neither to reads cause read
     * requests, nor does it buffer only a finite size.
     * It is sad that it is so difficult to simply get a program output.
     * (I don't really understand any of these issues. I would just use
     *  readline on exec's InputStream like it's done in CudaPath ... )
     */
    private static class StreamEater implements Runnable
    {
        private List<String>     m_stream     ;
        private InputStream      m_inputStream;
        private BufferedReader   m_reader     ;
        private volatile boolean m_done       ;

        public StreamEater ( InputStream input_stream )
        {
            m_inputStream = input_stream;
            m_reader      = new BufferedReader( new InputStreamReader( m_inputStream ) );
            m_stream      = new LinkedList<String>();
            m_done        = false;
            new Thread( this ).start(); // fork, calls method 'run'
        }

        public void run()
        {
            try
            {
                /* read all lines of the program and save into simple
                 * array of Strings */
                while ( true )
                {
                    String line = m_reader.readLine();
                    if ( line == null )
                        break;
                    m_stream.add( line );
                }
            }
            catch ( Exception ex ) {}
            finally {
                /* if not done with finally it could result in an infinite
                 * loop if this forked thread has an exception */
                m_done = true;
            }
        }

        public List<String> get()
        {
            while ( ! m_done )
            {
                try
                {
                    Thread.sleep(10);
                }
                catch ( Exception ex )
                {
                    ex.printStackTrace();
                }
            }
            return m_stream;
        }
    }
}
