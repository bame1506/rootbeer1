/*
 * Decompiled with CFR 0_121.
 *
 * Could not load the following classes:
 *  soot.rbclassload.HierarchyGraph
 *  soot.rbclassload.HierarchySignature
 *  soot.rbclassload.HierarchySootClass
 *  soot.rbclassload.HierarchySootMethod
 *  soot.rbclassload.MethodSignatureUtil
 *  soot.rbclassload.MultiDimensionalArrayTypeCreator
 *  soot.rbclassload.NumberedType
 *  soot.rbclassload.NumberedTypeSorter
 *  soot.rbclassload.RootbeerClassLoader
 *  soot.rbclassload.StringNumbers
 *  soot.rbclassload.StringToType
 */
package soot.rbclassload;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import soot.rbclassload.HierarchyGraph;
import soot.rbclassload.HierarchySignature;
import soot.rbclassload.HierarchySootClass;
import soot.rbclassload.HierarchySootMethod;
import soot.rbclassload.MethodSignatureUtil;
import soot.rbclassload.MultiDimensionalArrayTypeCreator;
import soot.rbclassload.NumberedType;
import soot.rbclassload.NumberedTypeSorter;
import soot.rbclassload.RootbeerClassLoader;
import soot.rbclassload.StringNumbers;
import soot.rbclassload.StringToType;

public class ClassHierarchy {
  private Map<Integer, HierarchySootClass> m_hierarchySootClasses = new TreeMap<Integer, HierarchySootClass>();
  private HierarchyGraph m_hierarchyGraph;
  private Map<HierarchySignature, List<HierarchySignature>> m_virtualMethodSignatures = new HashMap<HierarchySignature, List<HierarchySignature>>();
  private Set<Integer> m_leafs;
  private Set<Integer> m_interfaces;
  private Set<Integer> m_arrayTypes = new TreeSet<Integer>();
  private List<NumberedType> m_numberedTypes = new ArrayList<NumberedType>();
  private Map<String, NumberedType> m_numberedTypeMap = new HashMap<String, NumberedType>();
  private MethodSignatureUtil m_util = new MethodSignatureUtil();
  private int m_ifaceCount;
  private Set<Integer> m_visited;

  public void put(int name, HierarchySootClass hierarchy_class) {
    this.m_hierarchySootClasses.put(name, hierarchy_class);
  }

  public HierarchySootClass getHierarchySootClass(int name) {
    return this.m_hierarchySootClasses.get(name);
  }

  public HierarchySootClass getHierarchySootClass(String name) {
    Integer number = StringNumbers.v().addString(name);
    return this.m_hierarchySootClasses.get(number);
  }

  public Set<Integer> getClasses() {
    return this.m_hierarchySootClasses.keySet();
  }

  public Collection<HierarchySootClass> getHierarchyClasses() {
    return this.m_hierarchySootClasses.values();
  }

  public HierarchySootMethod getHierarchySootMethod(String signature) {
    this.m_util.parse(signature);
    String class_name = this.m_util.getClassName();
    HierarchySootClass hclass = this.getHierarchySootClass(class_name);
    if (hclass == null) {
      return null;
    }
    return hclass.findMethodBySubSignature(this.m_util.getSubSignature());
  }

  public HierarchySootMethod getHierarchySootMethod(HierarchySignature signature) {
    HierarchySootClass hclass = this.getHierarchySootClass(signature.getClassName());
    if (hclass == null) {
      return null;
    }
    return hclass.findMethodBySubSignature(signature);
  }

  public boolean containsClass(String name) {
    Integer number = StringNumbers.v().addString(name);
    return this.m_hierarchySootClasses.containsKey(number);
  }

  public void build() {
    HierarchyGraph hgraph;
    this.m_leafs = new HashSet<Integer>();
    this.m_interfaces = new HashSet<Integer>();
    this.m_leafs.addAll(this.m_hierarchySootClasses.keySet());
    for (Integer class_name : this.m_hierarchySootClasses.keySet()) {
      HierarchySootClass hsoot_class = this.m_hierarchySootClasses.get(class_name);
      this.m_leafs.remove(hsoot_class.getSuperClassNumber());
      for (Integer iface : hsoot_class.getInterfaceNumbers()) {
        this.m_leafs.remove(iface);
        this.m_interfaces.add(iface);
      }
    }
    this.m_hierarchyGraph = hgraph = new HierarchyGraph();
    for (Integer leaf : this.m_leafs) {
      LinkedList<Integer> queue = new LinkedList<Integer>();
      queue.add(leaf);
      while (!queue.isEmpty()) {
        Integer super_class;
        Integer curr_class = (Integer)queue.removeFirst();
        hgraph.addHierarchyClass(curr_class);
        HierarchySootClass hclass = this.m_hierarchySootClasses.get(curr_class);
        if (hclass == null) continue;
        if (hclass.isInterface()) {
          if (hclass.getInterfaceNumbers().isEmpty()) {
            if (!hclass.hasSuperClass()) continue;
            super_class = hclass.getSuperClassNumber();
            hgraph.addSuperClass(curr_class, super_class);
            queue.add(super_class);
            continue;
          }
          for (Integer iface : hclass.getInterfaceNumbers()) {
            hgraph.addInterface(curr_class, iface);
            queue.add(iface);
          }
          continue;
        }
        if (hclass.hasSuperClass()) {
          super_class = hclass.getSuperClassNumber();
          hgraph.addSuperClass(curr_class, super_class);
          queue.add(super_class);
        }
        for (Integer iface : hclass.getInterfaceNumbers()) {
          hgraph.addInterface(curr_class, iface);
          queue.add(iface);
        }
      }
    }
  }

  public void addArrayType(Integer array_type) {
    this.m_arrayTypes.add(array_type);
  }

  public void buildArrayTypes() {
    System.out.println("building array types...");
    MultiDimensionalArrayTypeCreator creator = new MultiDimensionalArrayTypeCreator();
    this.m_arrayTypes = creator.createInteger(this.m_arrayTypes);
    HierarchyGraph hgraph = this.m_hierarchyGraph;
    for (Integer curr_class : this.m_arrayTypes) {
      hgraph.addHierarchyClass(curr_class);
      hgraph.addSuperClass(curr_class, Integer.valueOf(0));
    }
  }

  public HierarchySootMethod findMethod(HierarchySignature signature) {
    int class_num = signature.getClassName();
    StringToType string_to_type = new StringToType();
    LinkedList<Integer> queue = new LinkedList<Integer>();
    queue.add(class_num);
    while (!queue.isEmpty()) {
      Integer curr_name = (Integer)queue.removeFirst();
      HierarchySootClass hclass = this.getHierarchySootClass(curr_name);
      if (hclass == null) {
        String class_str = StringNumbers.v().getString(curr_name.intValue());
        if (string_to_type.isArrayType(class_str)) {
          queue.add(0);
          continue;
        }
        return null;
      }
      HierarchySootMethod hmethod = hclass.findMethodBySubSignature(signature);
      if (hmethod == null) {
        if (hclass.hasSuperClass()) {
          queue.add(hclass.getSuperClassNumber());
        }
        for (Integer iface : hclass.getInterfaceNumbers()) {
          queue.add(iface);
        }
        continue;
      }
      return hmethod;
    }
    return null;
  }

  public HierarchySootMethod findMethod(String signature) {
    return this.findMethod(new HierarchySignature(signature));
  }

  public long getNumberForType(String type) {
    if (this.m_numberedTypeMap.containsKey(type)) {
      return this.m_numberedTypeMap.get(type).getNumber();
    }
    return -1;
  }

  public NumberedType getNumberedType(String str) {
    if (this.m_numberedTypeMap.containsKey(str)) {
      return this.m_numberedTypeMap.get(str);
    }
    System.out.println("cannot find numbered type: " + str);
    Iterator<String> iter = this.m_numberedTypeMap.keySet().iterator();
    while (iter.hasNext()) {
      System.out.println("  " + iter.next());
    }
    try {
      throw new RuntimeException("");
    }
    catch (Exception ex) {
      ex.printStackTrace(System.out);
      System.exit(0);
      return null;
    }
  }

  public void numberBuiltInType(String name, int number) {
    NumberedType numbered_type = new NumberedType(name, (long)number);
    this.m_numberedTypes.add(numbered_type);
    this.m_numberedTypeMap.put(name, numbered_type);
  }

  public void numberTypes() {
    Set<Integer> children;
    HierarchySootClass hclass;
    Integer str_num;
    System.out.println("numbering types...");
    int number = 1;
    LinkedList<Integer> queue = new LinkedList<Integer>();
    TreeSet<Integer> visited = new TreeSet<Integer>();
    HierarchyGraph hgraph = this.m_hierarchyGraph;
    this.numberBuiltInType("java.lang.Object", 1);
    this.numberBuiltInType("boolean", 2);
    this.numberBuiltInType("byte", 3);
    this.numberBuiltInType("char", 4);
    this.numberBuiltInType("short", 5);
    this.numberBuiltInType("int", 6);
    this.numberBuiltInType("float", 7);
    this.numberBuiltInType("double", 8);
    this.numberBuiltInType("boolean[]", 9);
    this.numberBuiltInType("byte[]", 10);
    this.numberBuiltInType("char[]", 11);
    this.numberBuiltInType("short[]", 12);
    this.numberBuiltInType("int[]", 13);
    this.numberBuiltInType("float[]", 14);
    this.numberBuiltInType("double[]", 15);
    number = 16;
    this.m_ifaceCount = 0;
    queue.add(0);
    while (!queue.isEmpty()) {
      str_num = (Integer)queue.removeFirst();
      if (visited.contains(str_num)) continue;
      visited.add(str_num);
      hclass = this.getHierarchySootClass(str_num);
      if (hclass == null) continue;
      children = hgraph.getChildren(str_num);
      for (Integer child : children) {
        queue.add(child);
      }
      if (!hclass.isInterface()) continue;
      ++this.m_ifaceCount;
    }
    number += this.m_ifaceCount + 1;
    this.m_visited = new HashSet<Integer>();
    this.topoVisit(0);
    queue.add(0);
    visited.clear();
    while (!queue.isEmpty()) {
      NumberedType numbered_type;
      String curr_type;
      str_num = (Integer)queue.removeFirst();
      if (visited.contains(str_num)) continue;
      visited.add(str_num);
      hclass = this.getHierarchySootClass(str_num);
      if (hclass == null) {
        curr_type = StringNumbers.v().getString(str_num.intValue());
        if (!curr_type.contains("[]") || this.m_numberedTypeMap.containsKey(curr_type)) continue;
        numbered_type = new NumberedType(curr_type, (long)number);
        this.m_numberedTypes.add(numbered_type);
        this.m_numberedTypeMap.put(curr_type, numbered_type);
        ++number;
        continue;
      }
      if (hclass.isInterface()) continue;
      children = hgraph.getChildren(str_num);
      for (Integer child : children) {
        HierarchySootClass child_hclass = this.getHierarchySootClass(child);
        if (child_hclass == null) {
          String curr_type2 = StringNumbers.v().getString(child.intValue());
          if (!curr_type2.contains("[]")) continue;
          queue.add(child);
          continue;
        }
        if (child_hclass.isInterface()) continue;
        queue.add(child);
      }
      if (str_num == 0) continue;
      curr_type = StringNumbers.v().getString(str_num.intValue());
      numbered_type = new NumberedType(curr_type, (long)number);
      this.m_numberedTypes.add(numbered_type);
      this.m_numberedTypeMap.put(curr_type, numbered_type);
      ++number;
    }
    Collections.sort(this.m_numberedTypes, new NumberedTypeSorter());
    this.m_leafs.clear();
    this.m_interfaces.clear();
    this.m_arrayTypes.clear();
    this.m_visited.clear();
  }

  private void topoVisit(Integer iface_class) {
    if (this.m_visited.contains(iface_class)) {
      return;
    }
    this.m_visited.add(iface_class);
    HierarchyGraph hgraph = this.m_hierarchyGraph;
    Set<Integer> children = hgraph.getChildren(iface_class);
    for (Integer child : children) {
      HierarchySootClass hclass = this.getHierarchySootClass(child);
      if (hclass == null || !hclass.isInterface()) continue;
      this.topoVisit(child);
    }
    if (iface_class != 0) {
      String class_name = StringNumbers.v().getString(iface_class.intValue());
      NumberedType numbered_type = new NumberedType(class_name, (long)this.m_ifaceCount);
      this.m_numberedTypes.add(numbered_type);
      this.m_numberedTypeMap.put(class_name, numbered_type);
      --this.m_ifaceCount;
    }
  }

  public List<NumberedType> getNumberedTypes() {
    ArrayList<NumberedType> ret_copy = new ArrayList<NumberedType>();
    ret_copy.addAll(this.m_numberedTypes);
    return ret_copy;
  }

  public void cacheVirtualMethods() {
    List<HierarchySootMethod> methods;
    HierarchySootClass hclass;
    HierarchySignature curr_sig;
    Integer curr_class;
    HashMap<HierarchySootMethod, HierarchySootMethod> virt_map = new HashMap<HierarchySootMethod, HierarchySootMethod>();
    for (Integer base_class : this.getClasses()) {
      HierarchySootClass base_hclass;
      if (this.m_interfaces.contains(base_class) || (base_hclass = this.getHierarchySootClass(base_class)) == null) continue;
      List<HierarchySootMethod> base_methods = base_hclass.getMethods();
      block1 : for (HierarchySootMethod method : base_methods) {
        HierarchySootClass super_hclass = base_hclass;
        while (super_hclass.hasSuperClass() && (super_hclass = this.getHierarchySootClass(super_hclass.getSuperClassNumber())) != null) {
          List<HierarchySootMethod> super_methods = super_hclass.getMethods();
          for (HierarchySootMethod super_method : super_methods) {
            if (!method.covarientMatch(super_method)) continue;
            virt_map.put(method, super_method);
            continue block1;
          }
        }
      }
    }
    HashSet<HierarchySignature> visited_sigs = new HashSet<HierarchySignature>();
    LinkedList<Integer> bfs_queue = new LinkedList<Integer>();
    bfs_queue.addAll(this.m_leafs);
    while (!bfs_queue.isEmpty()) {
      curr_class = (Integer)bfs_queue.removeFirst();
      hclass = this.getHierarchySootClass(curr_class);
      if (hclass == null) continue;
      methods = hclass.getMethods();
      for (HierarchySootMethod method : methods) {
        curr_sig = method.getHierarchySignature();
        if (visited_sigs.contains((Object)curr_sig)) continue;
        visited_sigs.add(curr_sig);
        ArrayList<HierarchySignature> path = new ArrayList<HierarchySignature>();
        path.add(curr_sig);
        this.m_virtualMethodSignatures.put(curr_sig, path);
        HierarchySootMethod trace_method = method;
        while (virt_map.containsKey((Object)trace_method)) {
          HierarchySignature trace_sig = (trace_method = (HierarchySootMethod)virt_map.get((Object)trace_method)).getHierarchySignature();
          if (visited_sigs.contains((Object)trace_sig)) {
            List<HierarchySignature> extended_path = this.m_virtualMethodSignatures.get(trace_sig);// new ArrayList<>(this.m_virtualMethodSignatures.get(trace_sig));
            for(HierarchySignature path_sig : path)
              if(!extended_path.contains(path_sig))
                extended_path.add(0, path_sig);
            this.m_virtualMethodSignatures.put(trace_sig, extended_path);
          } else {
            visited_sigs.add(trace_sig);
            // Make a copy of path to prevent modifying paths already set.
            path = new ArrayList<>(path);
            path.add(trace_sig);
            this.m_virtualMethodSignatures.put(trace_sig, path);
          }
        }
      }
      if (!hclass.hasSuperClass()) continue;
      bfs_queue.add(hclass.getSuperClassNumber());
    }
    visited_sigs.clear();
    bfs_queue.addAll(this.m_leafs);
    while (!bfs_queue.isEmpty()) {
      curr_class = (Integer)bfs_queue.removeFirst();
      hclass = this.getHierarchySootClass(curr_class);
      if (hclass == null) continue;
      methods = hclass.getMethods();
      for (HierarchySootMethod method : methods) {
        curr_sig = method.getHierarchySignature();
        if (visited_sigs.contains((Object)curr_sig)) continue;
        visited_sigs.add(curr_sig);
        LinkedList<Integer> ifaceQueue = new LinkedList<>(hclass.getInterfaceNumbers());
        while(!ifaceQueue.isEmpty()) {
          Integer iface = ifaceQueue.removeFirst();
          HierarchySootMethod iface_method;
          HierarchySootClass iface_hclass = this.getHierarchySootClass(iface);
          if (iface_hclass == null) continue;
          ifaceQueue.addAll(iface_hclass.getInterfaceNumbers());
          if ((iface_method = iface_hclass.findMethodBySubSignature(method)) == null) continue;
          List<HierarchySignature> path = this.m_virtualMethodSignatures.get((Object)curr_sig);
          HierarchySignature iface_sig = iface_method.getHierarchySignature();
          if (this.m_virtualMethodSignatures.containsKey((Object)iface_sig)) {
            ArrayList<HierarchySignature> new_path = new ArrayList<HierarchySignature>();
            new_path.addAll(path);
            List<HierarchySignature> old_path = this.m_virtualMethodSignatures.get((Object)iface_sig);
            for (HierarchySignature element : old_path) {
              if (new_path.contains((Object)element)) continue;
              new_path.add(element);
            }
            this.m_virtualMethodSignatures.put(iface_sig, new_path);
            continue;
          }
          this.m_virtualMethodSignatures.put(iface_sig, path);
        }
      }
      if (!hclass.hasSuperClass()) continue;
      bfs_queue.add(hclass.getSuperClassNumber());
    }
  }

  private void mapPut(Map<Integer, List<HierarchyGraph>> temp_graphs, Integer curr_class, HierarchyGraph hgraph) {
    List graphs;
    if (temp_graphs.containsKey(curr_class)) {
      graphs = temp_graphs.get(curr_class);
    } else {
      graphs = new ArrayList();
      temp_graphs.put(curr_class, graphs);
    }
    graphs.add(hgraph);
  }

  private void mapPut(Map<String, List<HierarchyGraph>> temp_graphs, String curr_class, HierarchyGraph hgraph) {
    List graphs;
    if (temp_graphs.containsKey(curr_class)) {
      graphs = temp_graphs.get(curr_class);
    } else {
      graphs = new ArrayList();
      temp_graphs.put(curr_class, graphs);
    }
    graphs.add(hgraph);
  }

  public HierarchyGraph getHierarchyGraph() {
    return this.m_hierarchyGraph;
  }

  public List<HierarchySignature> getVirtualMethods(HierarchySignature signature) {
    ArrayList<HierarchySignature> ret = new ArrayList<HierarchySignature>();
    ret.add(signature);
    String method_name = StringNumbers.v().getString(signature.getMethodName());
    if (method_name.equals("<init>") || method_name.equals("<clinit>")) {
      return ret;
    }
    if (!this.m_hierarchyGraph.getAllClasses().contains(signature.getClassName())) {
      return ret;
    }
    if (!this.m_virtualMethodSignatures.containsKey((Object)signature)) {
      return ret;
    }
    Set new_invokes = RootbeerClassLoader.v().getNewInvokes();
    List<HierarchySignature> virt_sigs = this.m_virtualMethodSignatures.get((Object)signature);
    for (HierarchySignature virt_sig : virt_sigs) {
      if (!new_invokes.contains(virt_sig.getClassName()) || ret.contains((Object)virt_sig)) continue;
      ret.add(virt_sig);
    }
    return ret;
  }

  public List<String> getVirtualMethods(String signature) {
    MethodSignatureUtil util = new MethodSignatureUtil();
    util.parse(signature);
    String class_name = util.getClassName();
    Integer class_num = StringNumbers.v().addString(class_name);
    ArrayList<String> ret = new ArrayList<String>();
    ret.add(signature);
    if (util.getMethodName().equals("<init>") || util.getMethodName().equals("<clinit>")) {
      return ret;
    }
    if (!this.m_hierarchyGraph.getAllClasses().contains(class_num)) {
      return ret;
    }
    HierarchySignature hierarchy_sig = new HierarchySignature(util);
    Set new_invokes = RootbeerClassLoader.v().getNewInvokes();
    if (!this.m_virtualMethodSignatures.containsKey((Object)hierarchy_sig)) {
      return ret;
    }
    List<HierarchySignature> virt_sigs = this.m_virtualMethodSignatures.get((Object)hierarchy_sig);
    for (HierarchySignature virt_sig : virt_sigs) {
      String string_sig = virt_sig.toString();
      if (!new_invokes.contains(virt_sig.getClassName()) || ret.contains(string_sig)) continue;
      ret.add(string_sig);
    }
    return ret;
  }
}