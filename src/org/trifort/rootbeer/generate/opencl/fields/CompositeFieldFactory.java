/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.opencl.fields;


import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.trifort.rootbeer.generate.opencl.OpenCLClass;
import org.trifort.rootbeer.generate.opencl.OpenCLScene;

import soot.SootClass;
import soot.SootField;


public final class CompositeFieldFactory
{

    public static List<CompositeField> getCompositeFields
    (
        final OpenCLScene scene,
        final Map<String, OpenCLClass> classes
    )
    {
        List<TreeNode>       hierarchy = new ReverseClassHierarchy( scene, classes ).get();
        List<CompositeField> fields    = new ArrayList<CompositeField>();
        Set<String>          processNodeVisited = new HashSet<String>();

        for ( final TreeNode node : hierarchy )
        {
            CompositeField composite = new CompositeField();
            processNode( node, composite, processNodeVisited );
            if ( composite.getClasses().isEmpty() )
                composite.getClasses().add( node.getSootClass() );
            composite.sort();
            fields.add( composite );
        }

        return fields;
    }

    private static void processNode
    (
        final TreeNode       node,
        final CompositeField composite,
        final Set<String>    processNodeVisited
    )
    {
        OpenCLClass ocl_class = node.getOpenCLClass();
        List<OpenCLField> ref_fields = ocl_class.getInstanceRefFields();
        for ( final OpenCLField field : ref_fields )
            processNodeField( node, field, true, composite, processNodeVisited );
        List<OpenCLField> static_ref_fields = ocl_class.getStaticRefFields();
        for ( final OpenCLField field : static_ref_fields )
            processNodeField( node, field, true, composite, processNodeVisited );
        List<OpenCLField> non_ref_fields = ocl_class.getInstanceNonRefFields();
        for ( final OpenCLField field : non_ref_fields )
            processNodeField( node, field, false, composite, processNodeVisited );
        List<OpenCLField> static_non_ref_fields = ocl_class.getStaticNonRefFields();
        for(OpenCLField field : static_non_ref_fields)
            processNodeField( node, field, false, composite, processNodeVisited );
        for ( final TreeNode child : node.getChildren() )
            processNode( child, composite, processNodeVisited );
    }

    private static void processNodeField
    (
        final TreeNode       node     ,
              OpenCLField    field    ,
        final boolean        ref_field,
        final CompositeField composite,
        final Set<String>    processNodeVisited
    )
    {
        SootClass soot_class = node.getSootClass();
        SootField soot_field = field.getSootField();

        OpenCLField new_field = new OpenCLField(soot_field, soot_class);

        boolean isCloned;
        try {
            soot_class.getFieldByName( soot_field.getName() );
            isCloned = false;
        } catch ( Exception ex ) {
            isCloned = true;
        }

        if ( isCloned )
            new_field.setClone( field );
        else
        {
            soot_field = soot_class.getFieldByName(soot_field.getName());
            new_field = new OpenCLField(soot_field, soot_class);
            field = new_field;
        }

        final String hash = soot_field.toString();
        if ( ! processNodeVisited.contains( hash ) )
        {
            processNodeVisited.add( hash );
            if ( ref_field )
                composite.addRefField(new_field, soot_class);
            else
                composite.addNonRefField(new_field, soot_class);
        }

        if ( ! soot_field.isPrivate() )
        {
            for ( final TreeNode child : node.getChildren() )
                processNodeField( child, field, ref_field, composite, processNodeVisited );
        }
    }
}
