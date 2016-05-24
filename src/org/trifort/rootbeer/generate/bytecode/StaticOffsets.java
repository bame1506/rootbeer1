/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.bytecode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.trifort.rootbeer.generate.opencl.OpenCLClass;
import org.trifort.rootbeer.generate.opencl.OpenCLScene;
import org.trifort.rootbeer.generate.opencl.fields.CompositeField;
import org.trifort.rootbeer.generate.opencl.fields.OpenCLField;

import soot.RefType;
import soot.Scene;
import soot.SootClass;
import soot.Type;
import soot.rbclassload.RootbeerClassLoader;


public class StaticOffsets
{
    /* if m_offsetToFieldMap was only the inverse map to m_fieldToOffsetMap
     * e.g. for performance reasons, then why does one have OpenCLField
     * while the other uses SortableField ?? */
    private final Map<Integer    , SortableField     > m_offsetToFieldMap;
    private final Map<OpenCLField, Integer           > m_fieldToOffsetMap;
    private final Map<SootClass  , Integer           > m_classToOffsetMap;
    private final Map<SootClass  , List<OpenCLField> > m_staticFields    ;
    private int                                        m_endIndex        ;
    private int                                        m_lockStart       ;
    private int                                        m_zerosSize       ;

    public StaticOffsets()
    {
        m_offsetToFieldMap = new HashMap<Integer    , SortableField     >();
        m_fieldToOffsetMap = new HashMap<OpenCLField, Integer           >();
        m_classToOffsetMap = new HashMap<SootClass  , Integer           >();
        m_staticFields     = new HashMap<SootClass  , List<OpenCLField> >();
        m_endIndex         = 0;
        m_lockStart        = 0;
        m_zerosSize        = 0;
        buildMaps();
    }

    public OpenCLField getField(int index           ){ return m_offsetToFieldMap.get(index).m_field; }
    public int         getIndex(OpenCLField field   ){ return m_fieldToOffsetMap.get(field)        ; }
    public int         getIndex(SootClass soot_class){ return m_classToOffsetMap.get(soot_class)   ; }
    public int         getEndIndex()                 { return m_endIndex                           ; }

    public List<OpenCLField> getStaticFields( final SootClass soot_class )
    {
        List<OpenCLField> ret = m_staticFields.get( soot_class );
        if ( ret == null ) {
            ret = new ArrayList<OpenCLField>();
        }
        return ret;
    }

    /**
     * Collects all classes and saves their corresponding address in a map
     * Note: I think this is similar to what `nm -C` shows, i.e. to one
     *       object file / class a list of functions and variables.
     */
    private void buildMaps()
    {
        final List<CompositeField> composites = OpenCLScene.v().getCompositeFields();
        final Set <SortableField > sortable_fields = new HashSet<SortableField>();
        for ( final CompositeField composite : composites )
        {
            for ( final SootClass soot_class : composite.getClasses() )
            {
                final List<OpenCLField> refs    = composite.getRefFieldsByClass   (soot_class);
                final List<OpenCLField> nonrefs = composite.getNonRefFieldsByClass(soot_class);

                sortable_fields.addAll( convert( refs   , soot_class) );
                sortable_fields.addAll( convert( nonrefs, soot_class) );

                final List<OpenCLField> static_fields = new ArrayList<OpenCLField>();
                static_fields.addAll( staticFilter( refs    ) );
                static_fields.addAll( staticFilter( nonrefs ) );

                m_staticFields.put( soot_class, static_fields );
            }
        }
        /* why is a new object being created if it is overwritten by toArray
         * right after that anyway ?? */
        SortableField[] array = new SortableField[ sortable_fields.size() ];
        array = sortable_fields.toArray(array);
        Arrays.sort( array );

        int index = 0;
        for ( final SortableField field : array )
        {
            m_offsetToFieldMap.put( index, field );
            m_fieldToOffsetMap.put( field.m_field, index );
            index += field.m_field.getSize();
        }
        /* Align to 4 (why? I don't think these are bytes, so why the need to
         * align?) Actually ... this isn't alignment... What is this oO?
         * Note also that +=0 does do nothing, so no need for if ... */
        int leftover = index % 4;
        if ( leftover != 0 ) {
            index += leftover;
        }
        m_lockStart = index;
        Set<Type> types = RootbeerClassLoader.v().getDfsInfo().getDfsTypes();
        for(Type type : types){
          if(type instanceof RefType == false){
            continue;
          }
          RefType ref_type = (RefType) type;
          String curr = ref_type.getClassName();
          SootClass soot_class = Scene.v().getSootClass(curr);
          m_classToOffsetMap.put(soot_class, index);
          index += 4;
        }
        m_endIndex = index;
        int mod = m_endIndex % 16;
        m_zerosSize = 0;
        if(mod != 0){
          m_endIndex += (16 - mod);
          m_zerosSize += (16 - mod);
        }
        //give room for junk space. some reason 32 bytes of space are needed for
        //the synchronized tests to pass.
        m_endIndex  += 32;
        m_zerosSize += 32;
    }

    public int getZerosSize(){ return m_zerosSize; }

  private List<SortableField> convert(List<OpenCLField> fields, SootClass soot_class) {
    fields = staticFilter(fields);
    List<SortableField> ret = new ArrayList<SortableField>();
    for(OpenCLField field : fields){
      ret.add(new SortableField(field, soot_class));
    }
    return ret;
  }

  private List<OpenCLField> staticFilter(List<OpenCLField> fields){
    List<OpenCLField> ret = new ArrayList<OpenCLField>();
    for(OpenCLField field : fields){
      if(field.isInstance() == false){
        ret.add(field);
      }
    }
    return ret;
  }

  public int getClassSize() {
    return m_classToOffsetMap.size();
  }

  public int getLockStart() {
    return m_lockStart;
  }

  private class SortableField implements Comparable<SortableField> {
    public OpenCLField m_field;
    public SootClass m_sootClass;

    public SortableField(OpenCLField field, SootClass soot_class){
      m_field = field;
      m_sootClass = soot_class;
    }

    public int compareTo(SortableField o) {
      int this_size = m_field.getSize();
      int o_size = o.m_field.getSize();
      return Integer.valueOf(o_size).compareTo(Integer.valueOf(this_size));
    }

    @Override
    public boolean equals(Object other){
      if(other instanceof SortableField == false){
        return false;
      }
      SortableField rhs = (SortableField) other;
      if(m_field.getName().equals(rhs.m_field.getName()) == false){
        return false;
      }
      if(m_field.getSootField().getDeclaringClass().getName().equals(rhs.m_field.getSootField().getDeclaringClass().getName()) == false){
        return false;
      }
      return true;
    }

    @Override
    public int hashCode() {
      int hash = 3;
      hash = 23 * hash + (this.m_field.toString() != null ? this.m_field.toString().hashCode() : 0);
      return hash;
    }
  }
}
