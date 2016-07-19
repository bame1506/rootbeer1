/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import org.trifort.rootbeer.configuration.RootbeerPaths;

public class ResourceReader
{
    /**
     * Returns a file packed into the jar as a String
     */
    public static String getResource( String path ) throws IOException
    {
        InputStream is = ResourceReader.class.getResourceAsStream( path );
        StringBuilder ret = new StringBuilder();
        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
        while(true)
        {
            String line = reader.readLine();
            if(line == null)
                break;
            ret.append(line + "\n");
        }
        is.close();
        return ret.toString();
    }

    /**
     * Buffers the file data in the stream returned by getResourceAsStream
     * into a byte array.
     * @todo Automatically determine length?
     */
    public static byte[] getResourceArray
    (
        String jar_path,
        int length
    ) throws IOException
    {
        jar_path = jar_path.replace("\\", "/");
        InputStream is = ResourceReader.class.getResourceAsStream(jar_path);
        byte[] ret = new byte[length];
        int offset = 0;
        /* @todo Not sure, why a loop is necessary? is.read should return as
         * many bytes as specified and we specify so much, that it exactly
         * fills the buffer specified. Furthermore if specified length is too
         * large, the while loop can't be exited !??? */
        while( offset < length )
        {
            int thisLength = length - offset;
            int read = is.read( ret, offset, thisLength );
            offset += read;
        }
        return ret;
    }

    /**
     * Reads specified file into a list of byte arrays, even if file size
     * not specified, by concatenating byte arrays as long as necessary for
     * the input stream to end.
     * @todo Other way to get file size first and then allocate whole buffer?
     *       @see http://stackoverflow.com/questions/26155314/why-does-getresourceasstream-and-reading-file-with-fileinputstream-return-arra
     *       @see http://www.coderanch.com/t/277331/java-io/java/determining-size-file-classpath
     *   => use ByteArrayOutputStream instead of implementing something similar anew
     **/
    public static List<byte[]> getResourceArray
    (
        String jar_path
    ) throws IOException
    {
        InputStream is = ResourceReader.class.getResourceAsStream( jar_path );
        if ( is == null )
            throw new RuntimeException( "Could not find 'jar_path' in this jar!" );

        List<byte[]> ret = new ArrayList<byte[]>();
        /* exists if returned length is -1, i.e. InputStream end reached */
        while(true)
        {
            /* load into a constant size buffer */
            final byte[] buffer = new byte[32*1024];
            final int len = is.read( buffer );
            if ( len == -1 )
                break;

            /**
             * copy into a buffer which is of equal or less size, normally
             * it is only less for the last bay array in the list, except
             * if the file size is a multiple of 32*1024.
             * @todo Increase performance by only copying into small buffer
             *       if len is < 32*1024, or just use ByteArrayOutputStream
             *       to return a byte array instead of a list of byte arrays
            */
            final byte[] small_buffer = new byte[len];
            for( int i = 0; i < len; ++i )
                small_buffer[i] = buffer[i];

            ret.add(small_buffer);
        }
        is.close();
        return ret;
    }

    /**
     * extracts a file from inside the jar-archive to the local file system
     * @todo instead of using getResourceArray, simple "pipe" InputStream
     *       from getResourceAsStream to FileOutputStream, if possible ???
     */
    public static void writeToFile
    (
        String jar_filename,
        String dest_filename
    ) throws IOException
    {
        final List<byte[]>     bytes = getResourceArray( jar_filename );
        final FileOutputStream fout  = new FileOutputStream( dest_filename );

        for ( byte[] byte_array : bytes )
           fout.write( byte_array );

        fout.flush();
        fout.close();
    }
}
