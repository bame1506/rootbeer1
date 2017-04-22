/*
 * Decompiled with CFR 0_121.
 *
 * Could not load the following classes:
 *  soot.rbclassload.FieldSignatureUtil
 *  soot.rbclassload.HierarchyInstruction
 *  soot.rbclassload.HierarchySignature
 *  soot.rbclassload.HierarchySootClass
 *  soot.rbclassload.HierarchySootMethod
 *  soot.rbclassload.Operand
 *  soot.rbclassload.StringNumbers
 *  soot.rbclassload.StringToType
 */
package soot.rbclassload;

import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import soot.rbclassload.ClassHierarchy;
import soot.rbclassload.FieldSignatureUtil;
import soot.rbclassload.HierarchyInstruction;
import soot.rbclassload.HierarchySignature;
import soot.rbclassload.HierarchySootClass;
import soot.rbclassload.HierarchySootMethod;
import soot.rbclassload.Operand;
import soot.rbclassload.RootbeerClassLoader;
import soot.rbclassload.StringNumbers;
import soot.rbclassload.StringToType;

public class HierarchyValueSwitch {
    private Set<Integer> m_refTypes = new HashSet<Integer>();
    private Set<Integer> m_arrayTypes = new HashSet<Integer>();
    private Set<Integer> m_allTypes = new HashSet<Integer>();
    private Set<HierarchySignature> m_methodRefs = new HashSet<HierarchySignature>();
    private Set<String> m_fieldRefs = new HashSet<String>();
    private Set<Integer> m_instanceofs = new HashSet<Integer>();
    private Set<Integer> m_newInvokes = new HashSet<Integer>();
    private Set<Integer> m_hierarchyVisited = new HashSet<Integer>();
    private HierarchySootMethod m_method;

    public Set<Integer> getRefTypesInteger() {
        return this.m_refTypes;
    }

    public Set<Integer> getAllTypesInteger() {
        return this.m_allTypes;
    }

    public Set<Integer> getArrayTypesInteger() {
        return this.m_arrayTypes;
    }

    public Set<HierarchySignature> getMethodRefsHierarchy() {
        return this.m_methodRefs;
    }

    public Set<String> getFieldRefs() {
        return this.m_fieldRefs;
    }

    public Set<Integer> getInstanceOfsInteger() {
        return this.m_instanceofs;
    }

    public Set<Integer> getNewInvokesInteger() {
        return this.m_newInvokes;
    }

    public void run(HierarchySignature signature) {
        ClassHierarchy class_hierarchy = RootbeerClassLoader.v().getClassHierarchy();
        HierarchySootMethod method = class_hierarchy.getHierarchySootMethod(signature);
        if (method == null) {
            return;
        }
        this.m_method = method;
        HierarchySootClass hclass = method.getHierarchySootClass();

        // Add new invokes for super classe (e.g. in case there are references to super)
        if(hclass.hasSuperClass())
            m_newInvokes.add(hclass.getSuperClassNumber());

        if (!method.isConcrete()) {
            return;
        }
        this.addHierarchy(signature.getClassName());
        this.addSignature(method);
        List<HierarchyInstruction> instructions = method.getInstructions();
        for (HierarchyInstruction inst : instructions) {
            this.addInstruction(inst);
        }
        List<Integer> ex_types = method.getCodeAttrExTypesInteger();
        for (Integer ex_type : ex_types) {
            this.addHierarchy(ex_type);
        }
    }

    private void addHierarchy(Integer type) {
        if (this.m_hierarchyVisited.contains(type)) {
            return;
        }
        this.m_hierarchyVisited.add(type);
        ClassHierarchy class_hierarchy = RootbeerClassLoader.v().getClassHierarchy();
        LinkedList<Integer> hierarchy_queue = new LinkedList<Integer>();
        hierarchy_queue.add(type);
        while (!hierarchy_queue.isEmpty()) {
            HierarchySootClass curr_hclass;
            Integer class_name;
            Integer org_class_name = class_name = (Integer)hierarchy_queue.removeFirst();
            this.addRefType(class_name);
            StringToType str_to_type = new StringToType();
            String class_str = StringNumbers.v().getString(class_name.intValue());
            if (str_to_type.isArrayType(class_str)) {
                this.addArrayType(class_name);
                class_str = str_to_type.getBaseType(class_str);
                class_name = StringNumbers.v().addString(class_str);
                this.addRefType(class_name);
            }
            if (!str_to_type.isRefType(class_str) || (curr_hclass = class_hierarchy.getHierarchySootClass(class_name)) == null) continue;
            if (curr_hclass.hasSuperClass()) {
                hierarchy_queue.add(curr_hclass.getSuperClassNumber());
            }
            hierarchy_queue.addAll(curr_hclass.getInterfaceNumbers());
        }
    }

    private void addRefType(Integer type) {
        this.m_refTypes.add(type);
        this.m_allTypes.add(type);
    }

    private void addArrayType(Integer type) {
        this.m_arrayTypes.add(type);
        this.m_allTypes.add(type);
    }

    private void addSignature(HierarchySootMethod method) {
        this.addHierarchy(method.getReturnTypeInteger());
        for (Integer param : method.getParameterTypesInteger()) {
            this.addHierarchy(param);
        }
        for (Integer except : method.getExceptionTypesInteger()) {
            this.addHierarchy(except);
        }
    }

    private void addInstruction(HierarchyInstruction inst) {
        this.addInstructionName(inst);
        this.addInstructionOperands(inst);
    }

    private void addInstructionName(HierarchyInstruction inst) {
        String name = inst.getName();
        if (name.equals("anewarray")) {
            this.addNewInvoke(inst);
        } else if (name.equals("instanceof")) {
            this.addInstanceOf(inst);
        } else if (name.equals("multianewarray")) {
            this.addNewInvoke(inst);
        } else if (name.equals("newarray")) {
            this.addNewInvoke(inst);
        } else if (name.equals("new")) {
            this.addNewInvoke(inst);
        }
    }

    private void addInstructionOperands(HierarchyInstruction inst) {
        List<Operand> operands = inst.getOperands();
        for (Operand operand : operands) {
            String value = operand.getValue();
            String type = operand.getType();
            if (type.equals("class_ref")) {
                this.addHierarchy(StringNumbers.v().addString(value));
                continue;
            }
            if (type.equals("method_ref")) {
                HierarchySignature method_ref = new HierarchySignature(value);
                this.m_methodRefs.add(method_ref);
                this.addHierarchy(method_ref.getClassName());
                this.addHierarchy(method_ref.getReturnType());
                for (Integer param : method_ref.getParams()) {
                    this.addHierarchy(param);
                }
                continue;
            }
            if (!type.equals("field_ref")) continue;
            this.m_fieldRefs.add(value);
            FieldSignatureUtil util = new FieldSignatureUtil();
            util.parse(value);
            this.addHierarchy(StringNumbers.v().addString(util.getDeclaringClass()));
            this.addHierarchy(StringNumbers.v().addString(util.getType()));
        }
    }

    private void addNewInvoke(HierarchyInstruction inst) {
        List<Operand> operands = inst.getOperands();
        for (Operand operand : operands) {
            String value = operand.getValue();
            String type = operand.getType();
            if (!type.equals("class_ref")) continue;
            this.m_newInvokes.add(StringNumbers.v().addString(value));
        }
    }

    private void addInstanceOf(HierarchyInstruction inst) {
        List<Operand> operands = inst.getOperands();
        for (Operand operand : operands) {
            String value = operand.getValue();
            String type = operand.getType();
            if (!type.equals("class_ref")) continue;
            int num = StringNumbers.v().addString(value);
            this.m_instanceofs.add(num);
            this.addHierarchy(num);
        }
    }
}
