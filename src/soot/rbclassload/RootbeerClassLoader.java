/*
 * Decompiled with CFR 0_121.
 *
 * Could not load the following classes:
 *  soot.Body
 *  soot.G
 *  soot.MethodSource
 *  soot.Scene
 *  soot.Singletons
 *  soot.Singletons$Global
 *  soot.SootClass
 *  soot.SootField
 *  soot.SootMethod
 *  soot.SourceLocator
 *  soot.Type
 *  soot.coffi.CoffiMethodSource
 *  soot.coffi.HierarchySootClassFactory
 *  soot.rbclassload.ClassHierarchy
 *  soot.rbclassload.ClassTester
 *  soot.rbclassload.DfsInfo
 *  soot.rbclassload.FieldSignatureUtil
 *  soot.rbclassload.HierarchySignature
 *  soot.rbclassload.HierarchySootClass
 *  soot.rbclassload.HierarchySootMethod
 *  soot.rbclassload.HierarchyValueSwitch
 *  soot.rbclassload.MethodSignatureUtil
 *  soot.rbclassload.MethodTester
 *  soot.rbclassload.NumberedType
 *  soot.rbclassload.StringCallGraph
 *  soot.rbclassload.StringNumbers
 *  soot.rbclassload.StringToType
 *  soot.util.Chain
 */
package soot.rbclassload;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarInputStream;
import soot.Body;
import soot.G;
import soot.MethodSource;
import soot.Scene;
import soot.Singletons;
import soot.SootClass;
import soot.SootField;
import soot.SootMethod;
import soot.SourceLocator;
import soot.Type;
import soot.coffi.CoffiMethodSource;
import soot.coffi.HierarchySootClassFactory;
import soot.rbclassload.ClassHierarchy;
import soot.rbclassload.ClassTester;
import soot.rbclassload.DfsInfo;
import soot.rbclassload.FieldSignatureUtil;
import soot.rbclassload.HierarchySignature;
import soot.rbclassload.HierarchySootClass;
import soot.rbclassload.HierarchySootMethod;
import soot.rbclassload.HierarchyValueSwitch;
import soot.rbclassload.MethodSignatureUtil;
import soot.rbclassload.MethodTester;
import soot.rbclassload.NumberedType;
import soot.rbclassload.StringCallGraph;
import soot.rbclassload.StringNumbers;
import soot.rbclassload.StringToType;
import soot.util.Chain;

public class RootbeerClassLoader {
    private Map<String, DfsInfo> m_dfsInfos = new HashMap<String, DfsInfo>();
    private DfsInfo m_currDfsInfo;
    private Map<String, Set<String>> m_packageNameCache = new HashMap<String, Set<String>>();
    private ClassHierarchy m_classHierarchy = new ClassHierarchy();
    private List<String> m_sourcePaths;
    private Map<String, String> m_classToFilename = new HashMap<String, String>();
    private String m_tempFolder;
    private List<String> m_classPaths;
    private int m_loadedCount;
    private List<MethodTester> m_entryMethodTesters;
    private List<MethodTester> m_dontFollowMethodTesters;
    private List<MethodTester> m_followMethodTesters;
    private List<MethodTester> m_toSignaturesMethodTesters;
    private List<ClassTester> m_dontFollowClassTesters;
    private List<ClassTester> m_followClassTesters;
    private List<ClassTester> m_toSignaturesClassTesters;
    private Map<String, String> m_classRemappings;
    private Map<String, String> m_reverseClassRemappings;
    private Set<String> m_followMethods;
    private Set<String> m_toSignaturesMethods;
    private Set<String> m_followClasses;
    private Set<String> m_toSignaturesClasses;
    private Set<String> m_toHierarchyClasses;
    private List<String> m_loadFields;
    private List<Integer> m_appClasses;
    private List<HierarchySignature> m_entryPoints;
    private Set<String> m_visited;
    private String m_userJar;
    private Set<String> m_generatedMethods;
    private Set<Integer> m_newInvokes;
    private Set<Integer> m_refTypes;
    private Map<HierarchySignature, HierarchyValueSwitch> m_valueSwitchMap;
    private boolean m_loaded;
    private Set<Integer> m_cgVisitedClasses;
    private Set<HierarchySignature> m_cgVisitedMethods;
    private LinkedList<HierarchySignature> m_cgMethodQueue;

    public RootbeerClassLoader(Singletons.Global g) {
        String home = System.getProperty("user.home");
        File soot_folder = new File(home + File.separator + ".soot" + File.separator + "rbcl_cache");
        soot_folder.mkdirs();
        this.m_tempFolder = soot_folder.getAbsolutePath() + File.separator;
        this.m_loadedCount = 0;
        this.m_entryMethodTesters = new ArrayList<MethodTester>();
        this.m_dontFollowMethodTesters = new ArrayList<MethodTester>();
        this.m_followMethodTesters = new ArrayList<MethodTester>();
        this.m_toSignaturesMethodTesters = new ArrayList<MethodTester>();
        this.m_dontFollowClassTesters = new ArrayList<ClassTester>();
        this.m_followClassTesters = new ArrayList<ClassTester>();
        this.m_toSignaturesClassTesters = new ArrayList<ClassTester>();
        this.m_followMethods = new HashSet<String>();
        this.m_toSignaturesMethods = new HashSet<String>();
        this.m_followClasses = new HashSet<String>();
        this.m_toSignaturesClasses = new HashSet<String>();
        this.m_toHierarchyClasses = new HashSet<String>();
        this.m_classRemappings = new HashMap<String, String>();
        this.m_reverseClassRemappings = new HashMap<String, String>();
        this.m_loadFields = new ArrayList<String>();
        this.m_appClasses = new ArrayList<Integer>();
        this.m_userJar = null;
        this.m_generatedMethods = new HashSet<String>();
        this.m_newInvokes = new HashSet<Integer>();
        this.m_refTypes = new HashSet<Integer>();
        this.m_valueSwitchMap = new HashMap<HierarchySignature, HierarchyValueSwitch>();
        this.m_cgVisitedClasses = new HashSet<Integer>();
        this.m_cgVisitedMethods = new HashSet<HierarchySignature>();
        this.m_cgMethodQueue = new LinkedList();
        this.m_loaded = false;
        this.loadBuiltIns();
    }

    public static RootbeerClassLoader v() {
        return G.v().soot_rbclassload_RootbeerClassLoader();
    }

    public void loadBuiltIns() {
        this.addBasicClassHierarchy("java.lang.Object");
        this.addBasicClassSignatures("java.lang.Class");
        this.addBasicClassSignatures("java.lang.Void");
        this.addBasicClassSignatures("java.lang.Boolean");
        this.addBasicClassSignatures("java.lang.Byte");
        this.addBasicClassSignatures("java.lang.Character");
        this.addBasicClassSignatures("java.lang.Short");
        this.addBasicClassSignatures("java.lang.Integer");
        this.addBasicClassSignatures("java.lang.Long");
        this.addBasicClassSignatures("java.lang.Float");
        this.addBasicClassSignatures("java.lang.Double");
        this.addBasicClassHierarchy("java.lang.String");
        this.addBasicClassSignatures("java.lang.StringBuffer");
        this.addBasicClassHierarchy("java.lang.Error");
        this.addBasicClassSignatures("java.lang.AssertionError");
        this.addBasicClassSignatures("java.lang.Throwable");
        this.addBasicClassSignatures("java.lang.NoClassDefFoundError");
        this.addBasicClassHierarchy("java.lang.ExceptionInInitializerError");
        this.addBasicClassHierarchy("java.lang.RuntimeException");
        this.addBasicClassHierarchy("java.lang.ClassNotFoundException");
        this.addBasicClassHierarchy("java.lang.ArithmeticException");
        this.addBasicClassHierarchy("java.lang.ArrayStoreException");
        this.addBasicClassHierarchy("java.lang.ClassCastException");
        this.addBasicClassHierarchy("java.lang.IllegalMonitorStateException");
        this.addBasicClassHierarchy("java.lang.IndexOutOfBoundsException");
        this.addBasicClassHierarchy("java.lang.ArrayIndexOutOfBoundsException");
        this.addBasicClassHierarchy("java.lang.NegativeArraySizeException");
        this.addBasicClassHierarchy("java.lang.NullPointerException");
        this.addBasicClassHierarchy("java.lang.InstantiationError");
        this.addBasicClassHierarchy("java.lang.InternalError");
        this.addBasicClassHierarchy("java.lang.OutOfMemoryError");
        this.addBasicClassHierarchy("java.lang.StackOverflowError");
        this.addBasicClassHierarchy("java.lang.UnknownError");
        this.addBasicClassHierarchy("java.lang.ThreadDeath");
        this.addBasicClassHierarchy("java.lang.ClassCircularityError");
        this.addBasicClassHierarchy("java.lang.ClassFormatError");
        this.addBasicClassHierarchy("java.lang.IllegalAccessError");
        this.addBasicClassHierarchy("java.lang.IncompatibleClassChangeError");
        this.addBasicClassHierarchy("java.lang.LinkageError");
        this.addBasicClassHierarchy("java.lang.VerifyError");
        this.addBasicClassHierarchy("java.lang.NoSuchFieldError");
        this.addBasicClassHierarchy("java.lang.AbstractMethodError");
        this.addBasicClassHierarchy("java.lang.NoSuchMethodError");
        this.addBasicClassHierarchy("java.lang.UnsatisfiedLinkError");
        this.addBasicClassHierarchy("java.lang.Thread");
        this.addBasicClassHierarchy("java.lang.Runnable");
        this.addBasicClassHierarchy("java.lang.Cloneable");
        this.addBasicClassHierarchy("java.io.Serializable");
        this.addBasicClassHierarchy("java.lang.ref.Finalizer");
        this.addBasicClassHierarchy("java.lang.ref.FinalReference");
    }

    private void addBasicClassHierarchy(String class_name) {
        this.m_toHierarchyClasses.add(class_name);
    }

    private void addBasicClassSignatures(String class_name) {
        this.m_toSignaturesClasses.add(class_name);
    }

    public void addEntryMethodTester(MethodTester method_tester) {
        this.m_entryMethodTesters.add(method_tester);
    }

    public void addDontFollowMethodTester(MethodTester method_tester) {
        this.m_dontFollowMethodTesters.add(method_tester);
    }

    public void addFollowMethodTester(MethodTester method_tester) {
        this.m_followMethodTesters.add(method_tester);
    }

    public void addToSignaturesMethodTester(MethodTester method_tester) {
        this.m_toSignaturesMethodTesters.add(method_tester);
    }

    public void addDontFollowClassTester(ClassTester class_tester) {
        this.m_dontFollowClassTesters.add(class_tester);
    }

    public void addFollowClassTester(ClassTester class_tester) {
        this.m_followClassTesters.add(class_tester);
    }

    public void addToSignaturesClassTester(ClassTester class_tester) {
        this.m_toSignaturesClassTesters.add(class_tester);
    }

    public void addNewInvoke(String class_name) {
        Integer class_num = StringNumbers.v().addString(class_name);
        this.m_newInvokes.add(class_num);
    }

    public void addGeneratedMethod(String signature) {
        this.m_generatedMethods.add(signature);
    }

    public void addClassRemapping(String original_class, String new_class) {
        this.m_classRemappings.put(original_class, new_class);
        this.m_reverseClassRemappings.put(new_class, original_class);
    }

    public Map<String, String> getReverseClassRemappings() {
        return this.m_reverseClassRemappings;
    }

    public void loadField(String field_sig) {
        this.m_loadFields.add(field_sig);
    }

    public List<String> getAllAppClasses() {
        throw new RuntimeException();
    }

    public ClassHierarchy getClassHierarchy() {
        return this.m_classHierarchy;
    }

    public void setUserJar(String filename) {
        this.m_userJar = filename;
    }

    public void loadNecessaryClasses() {
        this.m_sourcePaths = SourceLocator.v().sourcePath();
        this.m_classPaths = SourceLocator.v().classPath();
        this.loadHierarchySootClasses();
        this.remapClasses();
        this.buildClassHierarchy();
        this.findEntryPoints();
        this.loadFollowsStrings();
        this.loadToSignaturesString();
        for (HierarchySignature entry : this.m_entryPoints) {
            DfsInfo dfs_info = new DfsInfo(entry.toString());
            this.m_dfsInfos.put(entry.toString(), dfs_info);
        }
        int prev_size = -1;
        while (prev_size != this.m_newInvokes.size()) {
            prev_size = this.m_newInvokes.size();
            for (HierarchySignature entry : this.m_entryPoints) {
                System.out.println("entry point: " + entry.toString());
                DfsInfo dfs_info = new DfsInfo(entry.toString());
                this.m_currDfsInfo = this.m_dfsInfos.get(entry.toString());
                this.loadStringCallGraph();
            }
        }
        this.m_classHierarchy.buildArrayTypes();
        this.m_classHierarchy.numberTypes();
        this.loadScene();
        Scene.v().loadDynamicClasses();
    }

    public void setLoaded() {
        this.m_loaded = true;
    }

    private void cgMethodQueueAdd(HierarchySignature signature) {
        this.m_cgMethodQueue.add(signature);
    }

    private void loadStringCallGraph() {
        String entry = this.m_currDfsInfo.getRootMethodSignature();
        System.out.println("loading forward string call graph for: " + entry + "...");
        this.m_cgVisitedMethods.clear();
        HierarchySignature entry_hsig = new HierarchySignature(entry);
        this.cgMethodQueueAdd(entry_hsig);
        Set reverse_reachables = this.m_currDfsInfo.getReverseReachables();
        this.m_currDfsInfo.getStringCallGraph().addEntryPoint(entry);
        HierarchySootClass entry_class = this.m_classHierarchy.getHierarchySootClass(entry_hsig.getClassName());
        List<HierarchySootMethod> entry_methods = entry_class.getMethods();
        for (HierarchySootMethod entry_method : entry_methods) {
            if (!entry_method.getName().equals("<init>")) continue;
            this.cgMethodQueueAdd(entry_method.getHierarchySignature());
            reverse_reachables.add(entry_method.getHierarchySignature());
        }
        this.m_newInvokes.add(entry_hsig.getClassName());
        List<HierarchySignature> follow_sigs = this.getFollowSignatures();
        for (HierarchySignature sig : follow_sigs) {
            this.cgMethodQueueAdd(sig);
        }
        this.processForwardStringCallGraphQueue();
        System.out.println("loading reverse string call graph for: " + entry + "...");
        if (reverse_reachables.isEmpty()) {
            reverse_reachables.add(new HierarchySignature(this.m_currDfsInfo.getRootMethodSignature()));
        }
        for (Integer class_num : this.m_appClasses) {
            if (this.dontFollowClass(class_num)) continue;
            HierarchySootClass hclass = this.m_classHierarchy.getHierarchySootClass(class_num.intValue());
            List<HierarchySootMethod> methods = hclass.getMethods();
            for (HierarchySootMethod method : methods) {
                HierarchySignature signature = method.getHierarchySignature();
                if (this.dontFollowMethod(signature)) continue;
                this.reverseStringGraphVisit(signature, reverse_reachables);
            }
        }
    }

    private void processForwardStringCallGraphQueue() {
        HierarchySignature main_sig = new HierarchySignature("<java.lang.Object: void main(java.lang.String[])>");
        while (!this.m_cgMethodQueue.isEmpty()) {
            HierarchySignature bfs_entry = this.m_cgMethodQueue.removeFirst();
            if (this.m_cgVisitedMethods.contains((Object)bfs_entry)) continue;
            this.m_cgVisitedMethods.add(bfs_entry);
            HierarchySootMethod hmethod = this.m_classHierarchy.findMethod(bfs_entry);
            if (hmethod == null) continue;
            this.m_currDfsInfo.getStringCallGraph().addSignature(bfs_entry.toString());
            this.m_currDfsInfo.getStringCallGraph().addSignature(hmethod.getSignature());
            if (this.dontFollow(bfs_entry)) continue;
            bfs_entry = hmethod.getHierarchySignature();
            List<HierarchySignature> virt_methods = this.m_classHierarchy.getVirtualMethods(bfs_entry);
            for (HierarchySignature signature : virt_methods) {
                this.cgMethodQueueAdd(signature);
            }
            if (!hmethod.isConcrete()) continue;
            HierarchyValueSwitch value_switch = this.getValueSwitch(bfs_entry);
            this.m_newInvokes.addAll(value_switch.getNewInvokesInteger());
            for (HierarchySignature dest_sig : value_switch.getMethodRefsHierarchy()) {
                this.m_currDfsInfo.getStringCallGraph().addEdge(bfs_entry.toString(), dest_sig.toString());
                this.cgMethodQueueAdd(dest_sig);
            }
            for (Integer array_type : value_switch.getArrayTypesInteger()) {
                this.m_classHierarchy.addArrayType(array_type);
            }
            Set<Integer> class_refs = value_switch.getRefTypesInteger();
            for (Integer class_ref : class_refs) {
                HierarchySootMethod clinit_method;
                this.loadHierarchy(class_ref, this.m_refTypes);
                HierarchySootClass clinit_class = this.m_classHierarchy.getHierarchySootClass(class_ref.intValue());
                if (clinit_class == null || (clinit_method = clinit_class.findMethodBySubSignature("void <clinit>()")) == null) continue;
                HierarchySignature clinit_sig = clinit_method.getHierarchySignature();
                this.cgMethodQueueAdd(clinit_sig);
            }
            if (bfs_entry.subsigMatch(main_sig)) {
                String class_str = StringNumbers.v().getString(bfs_entry.getClassName());
                HierarchySignature ctor_sig = new HierarchySignature("<" + class_str + ": void <init>()>");
                this.cgMethodQueueAdd(ctor_sig);
            }
            if (this.m_cgVisitedClasses.contains(bfs_entry.getClassName())) continue;
            this.m_cgVisitedClasses.add(bfs_entry.getClassName());
            HierarchySootClass hclass = this.m_classHierarchy.getHierarchySootClass(bfs_entry.getClassName());
            if (hclass == null) {
                System.out.println("  hclass == null: " + StringNumbers.v().getString(bfs_entry.getClassName()));
                continue;
            }
            List<HierarchySootMethod> methods = hclass.getMethods();
            for (HierarchySootMethod method : methods) {
                String name = method.getName();
                if (!name.equals("<clinit>")) continue;
                this.cgMethodQueueAdd(method.getHierarchySignature());
            }
        }
    }

    private void reverseStringGraphVisit(HierarchySignature method_sig, Set<HierarchySignature> reachable) {
        int clinit_num = StringNumbers.v().addString("<clinit>");
        HierarchyValueSwitch value_switch = this.getValueSwitch(method_sig);
        for (HierarchySignature dest_sig : value_switch.getMethodRefsHierarchy()) {
            if (this.dontFollow(dest_sig) || !reachable.contains((Object)dest_sig)) continue;
            this.m_currDfsInfo.getStringCallGraph().addEdge(method_sig.toString(), dest_sig.toString());
            reachable.add(method_sig);
            if (!this.dontFollow(method_sig)) {
                this.cgMethodQueueAdd(method_sig);
            }
            List<HierarchySignature> virt_methods = this.m_classHierarchy.getVirtualMethods(method_sig);
            for (HierarchySignature virt_method : virt_methods) {
                if (reachable.contains((Object)virt_method)) continue;
                reachable.add(virt_method);
                if (this.dontFollow(virt_method)) continue;
                this.reverseStringGraphVisit(virt_method, reachable);
            }
            HierarchySootClass hclass = this.m_classHierarchy.getHierarchySootClass(method_sig.getClassName());
            List<HierarchySootMethod> methods = hclass.getMethods();
            for (HierarchySootMethod method : methods) {
                HierarchySignature class_sig = method.getHierarchySignature();
                if (this.dontFollow(class_sig) || class_sig.getMethodName() != clinit_num) continue;
                this.cgMethodQueueAdd(class_sig);
                reachable.add(class_sig);
            }
        }
        this.processForwardStringCallGraphQueue();
    }

    private List<HierarchySignature> getFollowSignatures() {
        ArrayList<HierarchySignature> ret = new ArrayList<HierarchySignature>();
        for (String follow_signature : this.m_followMethods) {
            ret.add(new HierarchySignature(follow_signature));
        }
        for (String follow_class : this.m_followClasses) {
            HierarchySootClass follow_hclass = this.m_classHierarchy.getHierarchySootClass(follow_class);
            List<HierarchySootMethod> follow_methods = follow_hclass.getMethods();
            for (HierarchySootMethod follow_method : follow_methods) {
                ret.add(follow_method.getHierarchySignature());
            }
        }
        return ret;
    }

    private void loadHierarchy(Integer class_name, Set<Integer> classes) {
        LinkedList<Integer> queue = new LinkedList<Integer>();
        queue.add(class_name);
        while (!queue.isEmpty()) {
            Integer curr_type = (Integer)queue.removeFirst();
            classes.add(curr_type);
            HierarchySootClass hclass = this.m_classHierarchy.getHierarchySootClass(curr_type.intValue());
            if (hclass == null) continue;
            if (hclass.hasSuperClass()) {
                queue.add(hclass.getSuperClassNumber());
            }
            for (Integer iface : hclass.getInterfaceNumbers()) {
                queue.add(iface);
            }
        }
    }

    private Collection<String> convertCollectionToString(Collection<Integer> input) {
        ArrayList<String> ret = new ArrayList<String>();
        for (Integer num : input) {
            ret.add(StringNumbers.v().getString(num.intValue()));
        }
        return ret;
    }

    private Collection<Integer> convertCollectionToInteger(Collection<String> input) {
        ArrayList<Integer> ret = new ArrayList<Integer>();
        for (String str : input) {
            ret.add(StringNumbers.v().addString(str));
        }
        return ret;
    }

    private void loadScene() {
        String type_string;
        String class_name;
        HierarchySootMethod method;
        FieldSignatureUtil util;
        int i;
        System.out.println("loading scene...");
        System.out.println("finding hierarchy classes reachable from dfs walk...");
        HashSet<Integer> all_types = new HashSet<Integer>();
        for (DfsInfo dfs_info : this.m_dfsInfos.values()) {
            all_types.addAll(this.convertCollectionToInteger(dfs_info.getStringCallGraph().getAllTypes()));
        }
        all_types.addAll(this.convertCollectionToInteger(this.m_toHierarchyClasses));
        all_types.addAll(this.convertCollectionToInteger(this.m_toSignaturesClasses));
        all_types.addAll(this.m_refTypes);
        HashSet<Integer> all_classes = new HashSet<Integer>();
        HashSet<String> visited_classes = new HashSet<String>();
        for (Integer type : all_types) {
            this.loadHierarchy(type, all_classes);
        }
        HashSet<String> all_sigs = new HashSet();
        for (DfsInfo dfs_info : this.m_dfsInfos.values()) {
            all_sigs.addAll(dfs_info.getStringCallGraph().getAllSignatures());
        }
        HashSet<String> to_signatures = new HashSet<String>();
        for (String signature_class : this.m_toSignaturesClasses) {
            HierarchySootClass signature_hclass = this.m_classHierarchy.getHierarchySootClass(signature_class);
            if (signature_hclass == null) {
                System.out.println("cannot find: " + signature_class);
                continue;
            }
            List<HierarchySootMethod> signature_methods = signature_hclass.getMethods();
            for (HierarchySootMethod signature_method : signature_methods) {
                to_signatures.add(signature_method.getSignature());
            }
        }
        to_signatures.addAll(all_sigs);
        to_signatures.addAll(this.m_toSignaturesMethods);
        for (String signature : to_signatures) {
            HierarchySootMethod method2;
            HierarchySignature hsig = new HierarchySignature(signature);
            HierarchySootClass hclass = this.m_classHierarchy.getHierarchySootClass(hsig.getClassName());
            if (hclass == null || (method2 = hclass.findMethodBySubSignature(hsig)) == null) continue;
            this.loadHierarchy(hsig.getClassName(), all_classes);
            this.loadHierarchy(hsig.getReturnType(), all_classes);
            for (Integer param_type : hsig.getParams()) {
                this.loadHierarchy(param_type, all_classes);
            }
            for (Integer ex_type : method2.getExceptionTypesInteger()) {
                this.loadHierarchy(ex_type, all_classes);
            }
        }
        System.out.println("creating empty classes according to type number...");
        StringToType string_to_type = new StringToType();
        List numbered_types = this.m_classHierarchy.getNumberedTypes();
        visited_classes.clear();
        for (i = 0; i < numbered_types.size(); ++i) {
            int type_num;
            type_string = ((NumberedType)numbered_types.get(i)).getType();
            if (!string_to_type.isRefType(type_string) || string_to_type.isArrayType(type_string) || visited_classes.contains(type_string) || !all_classes.contains(type_num = StringNumbers.v().addString(type_string))) continue;
            visited_classes.add(type_string);
            HierarchySootClass hclass = this.m_classHierarchy.getHierarchySootClass(type_string);
            SootClass empty_class = new SootClass(type_string, hclass.getModifiers());
            Scene.v().addClass(empty_class);
            if (hclass.isApplicationClass()) {
                empty_class.setApplicationClass();
            } else {
                empty_class.setLibraryClass();
            }
            if (hclass.hasSuperClass()) {
                SootClass superClass = Scene.v().getSootClass(hclass.getSuperClass());
                empty_class.setSuperclass(superClass);
            }
            for (String iface : hclass.getInterfaces()) {
                SootClass ifaceClass = Scene.v().getSootClass(iface);
                empty_class.addInterface(ifaceClass);
            }
        }
        for (i = 0; i < numbered_types.size(); ++i) {
            type_string = ((NumberedType)numbered_types.get(i)).getType();
            if (!string_to_type.isArrayType(type_string)) continue;
            SootClass empty_class = new SootClass(type_string, 1);
            Scene.v().addClass(empty_class);
            SootClass superClass = Scene.v().getSootClass("java.lang.Object");
            empty_class.setSuperclass(superClass);
        }
        System.out.println("filling in outer classes...");
        Chain chain = Scene.v().getClasses();
        SootClass curr = (SootClass)chain.getFirst();
        while (curr != null) {
            String name = curr.getName();
            if (name.contains("$")) {
                int index = name.lastIndexOf(36);
                String outer_class_str = name.substring(0, index);
                SootClass outer_class = Scene.v().getSootClass(outer_class_str);
                curr.setOuterClass(outer_class);
            }
            curr = (SootClass)chain.getSuccOf((Object)curr);
        }
        System.out.println("collecting fields for classes and adding to declaring class...");
        HashSet<String> fields_to_load = new HashSet<String>();
        HashSet<String> visited = new HashSet<String>();
        for (String signature : all_sigs) {
            HierarchySootMethod method3 = this.m_classHierarchy.findMethod(signature);
            if (method3 == null || visited.contains(method3.getSignature())) continue;
            visited.add(method3.getSignature());
            HierarchySignature hsig = new HierarchySignature(method3.getSignature());
            HierarchyValueSwitch value_switch = this.getValueSwitch(hsig);
            for (String field_ref : value_switch.getFieldRefs()) {
                fields_to_load.add(field_ref);
            }
        }
        fields_to_load.addAll(this.m_loadFields);
        block14 : for (String field_ref : fields_to_load) {
            HierarchySootClass hclass;
            util = new FieldSignatureUtil();
            util.parse(field_ref);
            class_name = util.getDeclaringClass();
            String field_name = util.getName();
            while ((hclass = this.m_classHierarchy.getHierarchySootClass(class_name)) != null) {
                if (!hclass.hasField(field_name)) {
                    if (!hclass.hasSuperClass()) {
                        System.out.println("cannot find field: " + field_ref);
                        continue block14;
                    }
                    class_name = hclass.getSuperClass();
                    continue;
                }
                SootClass declaring_class = Scene.v().getSootClass(class_name);
                if (declaring_class.declaresFieldByName(field_name)) continue block14;
                int field_modifiers = hclass.getFieldModifiers(field_name);
                Type field_type = string_to_type.convert(util.getType());
                SootField new_field = new SootField(field_name, field_type, field_modifiers);
                declaring_class.addField(new_field);
                continue block14;
            }
        }
        System.out.println("adding empty methods...");
        visited.clear();
        for (String signature : to_signatures) {
            MethodSignatureUtil methodUtil = new MethodSignatureUtil();
            methodUtil.parse(signature);
            class_name = methodUtil.getClassName();
            method = this.m_classHierarchy.findMethod(methodUtil.getSignature());
            if (method == null || visited.contains(method.getSignature())) continue;
            visited.add(method.getSignature());
            ArrayList<Type> parameterTypes = new ArrayList<Type>();
            for (String paramType : method.getParameterTypes()) {
                parameterTypes.add(string_to_type.convert(paramType));
            }
            Type returnType = string_to_type.convert(method.getReturnType());
            int modifiers = method.getModifiers();
            ArrayList<SootClass> thrownExceptions = new ArrayList<SootClass>();
            for (String exception : method.getExceptionTypes()) {
                SootClass ex_class = Scene.v().getSootClass(exception);
                thrownExceptions.add(ex_class);
            }
            SootMethod soot_method = new SootMethod(method.getName(), parameterTypes, returnType, modifiers, thrownExceptions);
            soot_method.setSource((MethodSource)method.getMethodSource());
            methodUtil.parse(method.getSignature());
            class_name = methodUtil.getClassName();
            SootClass soot_class = Scene.v().getSootClass(class_name);
            soot_class.addMethod(soot_method);
        }
        System.out.println("adding method bodies...");
        visited.clear();
        for (String signature : all_sigs) {
            SootMethod soot_method;
            MethodSignatureUtil methodUtil = new MethodSignatureUtil();
            methodUtil.parse(signature);
            class_name = methodUtil.getClassName();
            method = this.m_classHierarchy.findMethod(methodUtil.getSignature());
            if (method == null || visited.contains(method.getSignature())) continue;
            visited.add(method.getSignature());
            methodUtil.parse(method.getSignature());
            class_name = methodUtil.getClassName();
            SootClass soot_class = Scene.v().getSootClass(class_name);
            if (!soot_class.declaresMethod(method.getSubSignature()) || !(soot_method = soot_class.getMethod(method.getSubSignature())).isConcrete()) continue;
            soot_method.retrieveActiveBody();
        }
        System.out.println("Total loaded classes: " + all_classes.size());
        System.out.println("Total loaded methods: " + all_sigs.size());
    }

    public HierarchyValueSwitch getValueSwitch(HierarchySignature signature) {
        if (this.m_valueSwitchMap.containsKey((Object)signature)) {
            return this.m_valueSwitchMap.get((Object)signature);
        }
        HierarchyValueSwitch value_switch = new HierarchyValueSwitch();
        value_switch.run(signature);
        this.m_valueSwitchMap.put(signature, value_switch);
        return value_switch;
    }

    private void sortApplicationClasses() {
    }

    public boolean isLoaded() {
        return this.m_loaded;
    }

    public Set<Integer> getNewInvokes() {
        return this.m_newInvokes;
    }

    public List<SootMethod> getEntryPoints() {
        ArrayList<SootMethod> ret = new ArrayList<SootMethod>();
        for (HierarchySignature entry : this.m_entryPoints) {
            System.out.println("getEntryPoints: " + (Object)entry);
            MethodSignatureUtil util = new MethodSignatureUtil();
            util.parse(entry.toString());
            ret.add(util.getSootMethod());
        }
        return ret;
    }

    public void loadDfsInfo(SootMethod entry) {
        String sig = entry.getSignature();
        this.m_currDfsInfo = this.m_dfsInfos.get(sig);
    }

    private String normalizePathElement(String path) {
        if (File.separator.equals("/")) {
            return path;
        }
        if (path.startsWith("/")) {
            path = path.substring(1);
        }
        path = path.replace("/", "\\");
        return path;
    }

    private void loadHierarchySootClasses() {
        HierarchySootClassFactory hclassFactory = new HierarchySootClassFactory();
        List paths = SourceLocator.v().classPath();
        ArrayList<String> local_paths = new ArrayList<String>();
        local_paths.addAll(paths);
        this.m_userJar = this.normalizePathElement(this.m_userJar);
        if (this.m_userJar != null) {
            local_paths.add(this.m_userJar);
        }
        String[] to_cache = new String[local_paths.size()];
        to_cache = local_paths.toArray(to_cache);
        HashSet<String> visited = new HashSet<String>();
        for (String jar : to_cache) {
            File file;
            if (!(jar = this.normalizePathElement(jar)).endsWith(".jar") || !(file = new File(jar)).exists() || visited.contains(jar)) continue;
            visited.add(jar);
            System.out.println("caching package names for: " + jar);
            try {
                JarEntry entry;
                JarInputStream jin = new JarInputStream(new FileInputStream(jar));
                while ((entry = jin.getNextJarEntry()) != null) {
                    Set jars;
                    String package_name;
                    String name = entry.getName();
                    if (name.endsWith(".class")) {
                        String filename = name;
                        name = name.replace(".class", "");
                        name = name.replace("/", ".");
                        package_name = this.getPackageName(name);
                        boolean app_class = false;
                        if (jar.equals(this.m_userJar)) {
                            app_class = true;
                            this.m_appClasses.add(StringNumbers.v().addString(name));
                        }
                        HierarchySootClass hierarchy_class = hclassFactory.create((InputStream)jin, name);
                        hierarchy_class.setApplicationClass(app_class);
                        this.m_classHierarchy.put(hierarchy_class.getClassNumber(), hierarchy_class);
                    } else {
                        name = name.replace("/", ".");
                        package_name = name.substring(0, name.length() - 1);
                    }
                    if (this.m_packageNameCache.containsKey(package_name)) {
                        jars = this.m_packageNameCache.get(package_name);
                        if (jars.contains(jar)) continue;
                        jars.add(jar);
                        continue;
                    }
                    jars = new HashSet<String>();
                    jars.add((String)jar);
                    this.m_packageNameCache.put(package_name, jars);
                }
                jin.close();
            }
            catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    private void remapClasses() {
        for (String original_class : this.m_classRemappings.keySet()) {
            System.out.println("remapping class: " + original_class);
            String new_class = this.m_classRemappings.get(original_class);
            HierarchySootClass original_hclass = this.m_classHierarchy.getHierarchySootClass(original_class);
            HierarchySootClass new_hclass = this.m_classHierarchy.getHierarchySootClass(new_class);
            new_hclass.setName(original_class);
            new_hclass.setApplicationClass(original_hclass.isApplicationClass());
            this.m_classHierarchy.put(original_hclass.getClassNumber(), new_hclass);
        }
    }

    private void buildClassHierarchy() {
        System.out.println("building class hierarchy...");
        this.m_classHierarchy.build();
        System.out.println("caching virtual methods...");
        this.m_classHierarchy.cacheVirtualMethods();
    }

    private void loadFollowsStrings() {
        System.out.println("loading follows strings...");
        this.loadMethodStrings(this.m_followMethodTesters, this.m_followMethods);
        this.loadClassStrings(this.m_followClassTesters, this.m_followClasses);
    }

    private void loadToSignaturesString() {
        System.out.println("loading to_signatures strings...");
        this.loadMethodStrings(this.m_toSignaturesMethodTesters, this.m_toSignaturesMethods);
        this.loadClassStrings(this.m_toSignaturesClassTesters, this.m_toSignaturesClasses);
    }

    private void loadMethodStrings(List<MethodTester> testers, Set<String> dest) {
        Collection<HierarchySootClass> classes = this.m_classHierarchy.getHierarchyClasses();
        for (MethodTester tester : testers) {
            for (HierarchySootClass hclass : classes) {
                List<HierarchySootMethod> methods = hclass.getMethods();
                for (HierarchySootMethod hmethod : methods) {
                    if (!tester.test(hmethod)) continue;
                    dest.add(hmethod.getSignature());
                }
            }
        }
    }

    private void loadClassStrings(List<ClassTester> testers, Set<String> dest) {
        Collection<HierarchySootClass> classes = this.m_classHierarchy.getHierarchyClasses();
        for (ClassTester tester : testers) {
            for (HierarchySootClass hclass : classes) {
                if (!tester.test(hclass)) continue;
                dest.add(hclass.getName());
            }
        }
    }

    private String getPackageName(String className) {
        String[] tokens = className.split("\\.");
        String ret = "";
        for (int i = 0; i < tokens.length - 1; ++i) {
            ret = ret + tokens[i];
            if (i >= tokens.length - 2) continue;
            ret = ret + ".";
        }
        return ret;
    }

    public int getClassNumber(SootClass soot_class) {
        return this.getClassNumber(soot_class.getName());
    }

    public int getClassNumber(String type_string) {
        NumberedType ntype = this.m_classHierarchy.getNumberedType(type_string);
        return (int)ntype.getNumber();
    }

    private void findEntryPoints() {
        System.out.println("finding entry points...");
        this.m_entryPoints = new ArrayList<HierarchySignature>();
        for (Integer app_class : this.m_appClasses) {
            HierarchySootClass hclass = this.m_classHierarchy.getHierarchySootClass(app_class.intValue());
            List<HierarchySootMethod> methods = hclass.getMethods();
            for (HierarchySootMethod method : methods) {
                if (!this.testMethod(this.m_entryMethodTesters, method)) continue;
                this.m_entryPoints.add(method.getHierarchySignature());
            }
        }
    }

    public boolean dontFollow(HierarchySignature signature) {
        if (this.dontFollowMethod(signature)) {
            return true;
        }
        if (this.dontFollowClass(signature.getClassName())) {
            return true;
        }
        return false;
    }

    private boolean dontFollowMethod(HierarchySignature signature) {
        HierarchySootMethod hmethod = this.getHierarchySootMethod(signature);
        if (hmethod == null) {
            return false;
        }
        return this.testMethod(this.m_dontFollowMethodTesters, hmethod);
    }

    private boolean dontFollowClass(Integer class_name) {
        HierarchySootClass hclass = this.getHierarchySootClass(class_name);
        if (hclass == null) {
            return false;
        }
        return this.testClass(this.m_dontFollowClassTesters, hclass);
    }

    private boolean testMethod(List<MethodTester> testers, HierarchySootMethod hmethod) {
        for (MethodTester tester : testers) {
            if (!tester.test(hmethod)) continue;
            return true;
        }
        return false;
    }

    private boolean testClass(List<ClassTester> testers, HierarchySootClass hclass) {
        for (ClassTester tester : testers) {
            if (!tester.test(hclass)) continue;
            return true;
        }
        return false;
    }

    private HierarchySootClass getHierarchySootClass(Integer class_name) {
        return this.m_classHierarchy.getHierarchySootClass(class_name.intValue());
    }

    private HierarchySootMethod getHierarchySootMethod(HierarchySignature signature) {
        HierarchySootClass hclass = this.getHierarchySootClass(signature.getClassName());
        if (hclass == null) {
            return null;
        }
        return hclass.findMethodBySubSignature(signature);
    }

    public void addSignaturesClass(String class_name) {
        this.m_toSignaturesClasses.add(class_name);
    }

    public DfsInfo getDfsInfo() {
        return this.m_currDfsInfo;
    }
}
