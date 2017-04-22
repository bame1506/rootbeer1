/*
 * Decompiled with CFR 0_121.
 */
package soot.rbclassload;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;

public class HierarchyGraph {
    private Map<Integer, Set<Integer>> m_parents = new HashMap<Integer, Set<Integer>>();
    private Map<Integer, Set<Integer>> m_children = new HashMap<Integer, Set<Integer>>();
    private Set<Integer> m_hierarchy = new HashSet<Integer>();

    public void addSuperClass(Integer base_class, Integer super_class) {
        this.addEdge(this.m_parents, base_class, super_class);
        this.addEdge(this.m_children, super_class, base_class);
    }

    public void addInterface(Integer base_class, Integer iface) {
        this.addEdge(this.m_parents, base_class, iface);
        this.addEdge(this.m_children, iface, base_class);
    }

    public void addHierarchyClass(Integer class_name) {
        if (!this.m_hierarchy.contains(class_name)) {
            this.m_hierarchy.add(class_name);
        }
    }

    private void addEdge(Map<Integer, Set<Integer>> map, Integer key, Integer value) {
        Set values;
        if (map.containsKey(key)) {
            values = map.get(key);
        } else {
            values = new HashSet();
            map.put(key, values);
        }
        values.add(value);
    }

    public Set<Integer> getChildren(Integer parent) {
        if (this.m_children.containsKey(parent)) {
            return this.m_children.get(parent);
        }
        return new HashSet<Integer>();
    }

    public Set<Integer> getDescendants(Integer parent) {
        Set<Integer> ret = new HashSet<>();
        for(int child : getChildren(parent))
        {
            ret.add(child);
            ret.addAll(getChildren(child));
        }
        return ret;
    }

    public Set<Integer> getParents(Integer child) {
        if (this.m_parents.containsKey(child)) {
            return this.m_parents.get(child);
        }
        return new HashSet<Integer>();
    }


    public Set<Integer> getAncestors(Integer child) {
        Set<Integer> ret = new HashSet<>();
        for(int parent : getParents(child))
        {
            ret.add(parent);
            ret.addAll(getParents(parent));
        }
        return ret;
    }

    public Set<Integer> getAllClasses() {
        return this.m_hierarchy;
    }

    public boolean sameHierarchy(int number1, int number2) {
        int curr;
        LinkedList<Integer> queue = new LinkedList<Integer>();
        queue.addAll(this.getParents(number1));
        while (!queue.isEmpty()) {
            curr = (Integer)queue.removeFirst();
            if (curr == number2) {
                return true;
            }
            queue.addAll(this.getParents(curr));
        }
        queue.addAll(this.getChildren(number1));
        while (!queue.isEmpty()) {
            curr = (Integer)queue.removeFirst();
            if (curr == number2) {
                return true;
            }
            queue.addAll(this.getChildren(curr));
        }
        return false;
    }

    public String toString() {
        StringBuilder ret = new StringBuilder();
        ret.append(this.printMap(this.m_parents, "m_parents", "child", "parent"));
        ret.append(this.printMap(this.m_children, "m_children", "parent", "child"));
        return ret.toString();
    }

    private String printMap(Map<Integer, Set<Integer>> map, String heading, String key_name, String value_name) {
        StringBuilder ret = new StringBuilder();
        ret.append(heading + "\n");
        for (Integer key : map.keySet()) {
            Set<Integer> values = map.get(key);
            ret.append("  " + key_name + ": " + key + " " + value_name + ": " + values.toString() + "\n");
        }
        return ret.toString();
    }
}