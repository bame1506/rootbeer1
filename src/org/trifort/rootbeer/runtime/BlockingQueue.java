/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime;

/**
 * @see https://docs.oracle.com/javase/7/docs/api/java/util/concurrent/LinkedBlockingQueue.html
 * A basic thread-safe FIFO queue with atomic methods for adding and removing
 */
import java.util.concurrent.LinkedBlockingQueue;

/**
 * This is a wrapper for LinkedBlockingQueue which retries put and take
 * until it succeeds, ignoring exceptions in the process.
 * Each kernel launch will be put into this queue.
 */
public class BlockingQueue<T>
{
    private LinkedBlockingQueue<T> m_Queue;

    public BlockingQueue(){
        m_Queue = new LinkedBlockingQueue<T>();
    }

    public void put(T item)
    {
        while(true){
          try {
            m_Queue.put(item);
            return;
          } catch (Exception ex){
            //continue;
          }
        }
    }

    public int size(){
        return m_Queue.size();
    }

    /**
     * Retrieves and removes the head of this queue, waiting if necessary
     * until an element becomes available.
     */
    public T take()
    {
        while(true){
          try {
            return m_Queue.take();
          } catch (Exception ex){
            //continue;
          }
        }
    }
}
