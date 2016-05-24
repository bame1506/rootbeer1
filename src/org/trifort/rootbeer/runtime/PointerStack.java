/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime;

import java.util.ArrayList;
import java.util.List;

/**
 * A stack which holds Long values. Is there really not standard library
 * for this in Java ?
 * E.g. what speaks against
 *    https://docs.oracle.com/javase/7/docs/api/java/util/Stack.html
 *      has: push, pop, peek, empty
 * ??? It seems like this exists since Java 1.5, so it shouldn't be
 * compatibility reasons
 * Only used by FixedMemory.java:Memory class for one variable ...
 */
public class PointerStack
{
  private final List<Long> m_Stack;
  private int              m_Index;
  private final int        m_DefaultDepth; /**< maximum size */

  public PointerStack()
  {
      m_DefaultDepth = 16;
      m_Stack = new ArrayList<Long>(m_DefaultDepth);
      m_Index = 0; /* not necessary .. why then omit it in FixedMemory, where it adds clarity! (adds clarity here too) */
      for ( int i = 0; i < m_DefaultDepth; ++i )
          m_Stack.add(0L);
  }

  public void push(long value)
  {
      m_Index++;
      if ( m_Index < m_DefaultDepth )
      {
          m_Stack.set(m_Index, value);
          return;
      }
      else
      {
          /* this while loop seems unnecesarily superfluous, because m_Index
           * is only incremented by exactly one and only in this method, meaning
           * it should never be the case to add more than one element!
           * Also this doesn't increase m_DefaultDepth which is confusing, but
           * may not result in a bug, because a stack doesn't allow
           * random access. */
          while ( m_Stack.size() <= m_Index ) {
              m_Stack.add(0L);
          }
          m_Stack.set(m_Index, value);
      }
  }

  public long pop()
  {
      long ret = m_Stack.get(m_Index);
      m_Index--;
      return ret;
  }
}
