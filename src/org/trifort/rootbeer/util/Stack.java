/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.util;


import java.util.ArrayList;
import java.util.List;


/**
 * @todo Why not simply use java.util.Stack which exists at least since 1.5.0?
 *       Because it has no top and size methods? For top there is peek
 *       And size is only used one single time at:
 *          generate/opencl/body/MonitorGroups.java:64:
 *             if(stack.size() == 0){
 *       Which can be replaced with empty()
 */
public class Stack<T>
{
    private List<T> m_Data;
    public      Stack(){ m_Data = new ArrayList<T>(); }
    public void pop  (){ m_Data.remove(m_Data.size()-1); }
    public T    top  (){ return m_Data.get(m_Data.size()-1); }
    public int  size (){ return m_Data.size(); }
    public void push(T value){ m_Data.add(value); }
}
