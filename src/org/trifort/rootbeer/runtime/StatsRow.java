/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime;

/* Saves all timings in seconds. this could have been a simple struct (POD) ... */
public class StatsRow
{
    private long driverMemcopyToDeviceTime;     /**< This is measured by CUDAContext.c in milliseconds */
    private long driverExecTime;                /**< This is measured by CUDAContext.c in milliseconds */
    private long driverMemcopyFromDeviceTime;   /**< This is measured by CUDAContext.c in milliseconds */
    private long serializationTime;             /**< Time to serialize from Java to GPU */
    private long totalDriverExecutionTime;      /**< Time to execute on GPU */
    private long deserializationTime;           /**< Time to deserialize from GPU to Java */
    private long overallTime;                   /**< Overall time ((de)serialization + exec + time in runtime) */
    /* Accessor boiler-plate */
    public StatsRow() {}
    public void setDriverTimes /* used by CUDAContext.c */
    (
        long memcopyToDevice,
        long execTime,
        long memcopyFromDevice
    )
    {
        driverMemcopyToDeviceTime   = memcopyToDevice;
        driverExecTime              = execTime;
        driverMemcopyFromDeviceTime = memcopyFromDevice;
    }
    public long getDriverMemcopyToDeviceTime  (){ return driverMemcopyToDeviceTime  ; }
    public long getDriverMemcopyFromDeviceTime(){ return driverMemcopyFromDeviceTime; }
    public long getDriverExecTime             (){ return driverExecTime             ; }
    public long getSerializationTime          (){ return serializationTime          ; }
    public long getTotalDriverExecutionTime   (){ return totalDriverExecutionTime   ; }
    public long getDeserializationTime        (){ return deserializationTime        ; }
    public long getOverallTime                (){ return overallTime                ; }
    public void setSerializationTime  ( long time ){ serializationTime        = time; }
    public void setExecutionTime      ( long time ){ totalDriverExecutionTime = time; }
    public void setDeserializationTime( long time ){ deserializationTime      = time; }
    public void setOverallTime        ( long time ){ overallTime              = time; }
}
