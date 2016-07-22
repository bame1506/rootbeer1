/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.entry;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

/**
 * The function signatures which are to be loaded
 */
public class ForcedFields
{
    private final static List<String> m_fields = Arrays.asList(
        "<java.lang.Boolean: boolean value>"                               ,
        "<java.lang.Integer: int value>"                                   ,
        "<java.lang.Long: long value>"                                     ,
        "<java.lang.Float: float value>"                                   ,
        "<java.lang.Double: double value>"                                 ,
        "<java.lang.Class: java.lang.String name>"                         ,
        "<java.lang.AbstractStringBuilder: char[] value>"                  ,
        "<java.lang.AbstractStringBuilder: int count>"                     ,
        "<org.trifort.rootbeer.runtimegpu.GpuException: int m_arrayLength>",
        "<org.trifort.rootbeer.runtimegpu.GpuException: int m_arrayIndex>" ,
        "<org.trifort.rootbeer.runtimegpu.GpuException: int m_array>"      ,
        "<org.trifort.rootbeer.runtime.GpuStopwatch: long m_start>"        ,
        "<org.trifort.rootbeer.runtime.GpuStopwatch: long m_stop>"
    );

    public static List<String> get()
    {
        return m_fields;
    }
}
