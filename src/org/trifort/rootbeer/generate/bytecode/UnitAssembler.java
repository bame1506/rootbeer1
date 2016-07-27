/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.bytecode;


import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import soot.Body;
import soot.Local;
import soot.PatchingChain;
import soot.Unit;
import soot.UnitBox;
import soot.Value;
import soot.ValueBox;

import soot.jimple.GotoStmt;
import soot.jimple.IfStmt;
import soot.jimple.Jimple;
import soot.util.Chain;


public class UnitAssembler
{
    private final List<Local>                m_outputLocals     ;
    private final Map<String, Local>         m_localMap         ;
    private final Map<String, List<UnitBox>> m_labelToUnitBoxMap;
    /* Unit can be Jimple.GotoSmt or Jimple.IfStmt and other Jimple statements */
    private final List<Unit>                 m_inputUnits       ;
    /* just a cloned version of m_inputUnits ??? */
    private final List<Unit>                 m_outputUnits      ;
    private final List<List<String>>         m_labels           ;
    private final Jimple                     m_jimple           ;

    public UnitAssembler()
    {
        m_outputUnits       = new ArrayList<Unit>();
        m_outputLocals      = new ArrayList<Local>();
        m_localMap          = new HashMap<String, Local>();
        m_labelToUnitBoxMap = new HashMap<String, List<UnitBox>>();
        m_inputUnits        = new ArrayList<Unit>();
        m_labels            = new ArrayList<List<String>>();
        m_jimple            = Jimple.v();
    }

    public void add( Unit u ){ m_inputUnits.add(u); }

    public void addAll(Collection<Unit> units)
    {
        for ( Unit u : units )
            m_inputUnits.add(u);
    }

    void copyLocals()
    {
        for ( final Unit u : m_outputUnits )
        {
            List<ValueBox> boxes = u.getUseAndDefBoxes();
            for(ValueBox box : boxes){
                Value v = box.getValue();
                if(v instanceof Local == false)
                    continue;
                Local local = (Local) v.clone();
                if(m_localMap.containsKey(local.toString()) == false){
                    m_localMap.put(local.toString(), local);
                    m_outputLocals.add(local);
                }
                local = m_localMap.get(local.toString());
                box.setValue(local);
            }
        }
    }

    UnitBox getTarget( Unit input )
    {
        if(input instanceof IfStmt){
            IfStmt if_stmt = (IfStmt) input;
            return if_stmt.getTargetBox();
        } else if(input instanceof GotoStmt){
            GotoStmt goto_stmt = (GotoStmt) input;
            return goto_stmt.getTargetBox();
        }
        return null;
    }

    void copyTargets()
    {
        for(int i = 0; i < m_inputUnits.size(); ++i){
            Unit input = m_inputUnits.get(i);
            Unit output = m_outputUnits.get(i);
            List<UnitBox> input_boxes = input.getUnitBoxes();
            List<UnitBox> output_boxes = output.getUnitBoxes();
            for(int j = 0; j < input_boxes.size(); ++j){
                UnitBox input_box = input_boxes.get(j);
                UnitBox output_box = output_boxes.get(j);

                Unit input_target = input_box.getUnit();
                //using the addIf method makes targets null
                if(input_target == null)
                    continue;

                int target_i = findTarget(input_target);
                output_box.setUnit(m_outputUnits.get(target_i));
            }
        }
    }

    public Unit unitClone(Unit input)
    {
        Unit output = (Unit) input.clone();
        List<UnitBox> input_boxes = input.getUnitBoxes();
        List<UnitBox> output_boxes = output.getUnitBoxes();
        for(int i = 0; i < input_boxes.size(); ++i){
            UnitBox input_box = input_boxes.get(i);
            UnitBox output_box = output_boxes.get(i);
            try {
                int j = findTarget(input_box.getUnit());
                output_box.setUnit(m_inputUnits.get(j));
            } catch(Exception ex){
                ex.printStackTrace();
                continue;
            }
        }
        return output;
    }

    private boolean unitEquals(Unit lhs, Unit rhs)
    {
        if ( lhs.equals(rhs) )
            return true;
        if ( lhs instanceof GotoStmt && rhs instanceof GotoStmt )
        {
            GotoStmt lhs_goto = (GotoStmt) lhs;
            GotoStmt rhs_goto = (GotoStmt) rhs;
            if(lhs_goto.getTarget().equals(rhs_goto.getTarget()))
                return true;
        }
        return false;
    }

    int findTarget(Unit target)
    {
        for(int i = 0; i < m_inputUnits.size(); ++i){
            Unit curr = m_inputUnits.get(i);
            if(unitEquals(target, curr))
                return i;
        }
        throw new RuntimeException("Cannot find target while assembling units: " + target.toString());
    }

    public void assemble(Body body)
    {
        assignLabels();
        cloneUnits();
        copyTargets();
        checkTargetBoxes();
        copyLocals();
        writeToBody(body);
    }

    /**
     * checks for undefined labeled code blocks.
     */
    void checkTargetBoxes()
    {
        final Set<String> key_set = m_labelToUnitBoxMap.keySet();
        for ( final String key : key_set )
        {
            /* get all boxes to the current label. Why are multple boxes
             * to one label allowed ? */
            final List<UnitBox> boxes = m_labelToUnitBoxMap.get( key );
            for ( final UnitBox box : boxes )
            {
                if ( box.getUnit() == null )
                    throw new RuntimeException( "box unit is null: " + key );
            }
        }
    }

    void cloneUnits()
    {
        for ( final Unit u : m_inputUnits )
            m_outputUnits.add( (Unit) u.clone() );
    }

    private void writeToBody( Body body )
    {
        final PatchingChain<Unit> units = body.getUnits();
        Chain<Local> locals = body.getLocals();
        units.clear();
        locals.clear();

        for ( Unit u : m_outputUnits )
            units.add(u);
        for ( Local l : m_outputLocals )
            locals.add(l);
    }

    /**
     * Only for debugging purposes. Output the generated Jimple source code
     * I.e. output List<Unit> m_outputUnits and possibly m_labels
     */
    @Override public String toString()
    {
        String ret = "";
        for ( int i = 0; i < m_outputUnits.size(); ++i )
        {
            if ( i < m_labels.size() )
            {
                /* Output all, possibly multiple labels associated with this
                 * code block */
                for ( final String label : m_labels.get(i) )
                    ret += label + ":\n";
            }

            /* The jimple goto statement seems bugged, resulting in output
             * like this:
             *
             *    if rbreg14 == 0
             *        goto rbreg17 = parameter0 instanceof long[]
             *    [...]
             *    rblabel0:
             *        rbreg17 = parameter0 instanceof long[]
             *    [...]
             *
             * I.e. instead of the goto label the first instruction after
             * it is printed as the target. Instead we manually force it
             * to:
             *
             *    goto rblabel0
             *
             * @todo doesn't find label and does recognize if ... goto
             */

            final Unit statement = m_outputUnits.get(i);
            //if ( statement instanceof GotoStmt ||
            //     ( statement instanceof IfStmt &&
            //       ( (IfStmt) statement ).getTarget() instanceof GotoStmt )
            //   )
            //{
            //    UnitBox jmpTarget;
            //    if ( statement instanceof GotoStmt )
            //        jmpTarget = ( (GotoStmt) statement ).getTargetBox();
            //    else
            //        jmpTarget = ( (GotoStmt) ( (IfStmt) statement ).getTarget() ).getTargetBox();
            //
            //    String foundLabel = null;
            //    /* Try to find the asociated UnitBox in HashMap m_labelToUnitBoxMap */
            //    for ( final String key : m_labelToUnitBoxMap.keySet() )
            //    {
            //        /* get all boxes to the current label. Why are multple boxes
            //         * to one label allowed ? */
            //        for ( final UnitBox box : m_labelToUnitBoxMap.get( key ) )
            //        {
            //            // diret comparison not working, because of cloning ???
            //            // if ( box == jmpTarget )
            //            /* comparing strings may lead to false positives !!! */
            //            if ( box.toString().equals( jmpTarget.toString() ) )
            //            {
            //                foundLabel = key;
            //                break;
            //            }
            //        }
            //        if ( foundLabel != null )
            //            break;
            //    }
            //    if ( foundLabel == null )
            //        foundLabel = "[LABEL NOT FOUND]";
            //
            //    if ( statement instanceof IfStmt )
            //    {
            //        ret += "if " + ( (IfStmt) statement ).getConditionBox().toString() +
            //               "\n    ";
            //    }
            //    ret += "goto " + foundLabel + "\n";
            //}
            //else
                ret += statement.toString() + "\n";
        }
        return ret;
    }

    /**
     * Searches the label in m_labelToUnitBoxMap.
     * If not found create a new list of UnitBoxes and add the given unit_box
     * A UnitBox is a block of instructions
     * If found add the UnitBox to the list corresponding to the label
     * @todo it seems wrong to add multiple unit_boxes to the same label here
     *       What of those should be executed first Oo?
     */
    private void addLabelToUnitBox
    (
        final String  label,
        final UnitBox unit_box
    )
    {
        List<UnitBox> boxes;
        if ( m_labelToUnitBoxMap.containsKey(label) )
            boxes = m_labelToUnitBoxMap.get(label);
        else
            boxes = new ArrayList<UnitBox>();

        boxes.add( unit_box );
        m_labelToUnitBoxMap.put( label, boxes );
    }

    /**
     * adds conditional jump to a label if condition is true
     */
    public void addIf( final Value condition, final String target_label )
    {
        final UnitBox target = m_jimple.newStmtBox( null );
        addLabelToUnitBox( target_label, target );
        /* if condition, then execute instruction 'target' */
        add( m_jimple.newIfStmt( condition, target ) );
    }

    public void addGoto( String target_label )
    {
        UnitBox target = m_jimple.newStmtBox( null );
        addLabelToUnitBox( target_label, target );
        Unit u = m_jimple.newGotoStmt( target );
        add(u);
    }

    /**
     * adds a label to the current inputUnit. Possibly padding the label list
     */
    public void addLabel( final String label )
    {
        /* pad m_labels with empty lists until same length as m_inputUnits */
        while ( m_inputUnits.size() >= m_labels.size() )
            m_labels.add( new ArrayList<String>() );
        /* add the label to the last list */
        m_labels.get( m_labels.size()-1 ).add( label );
    }

    public Unit getUnitByLabel(String label)
    {
        for ( int i = 0; i < m_labels.size(); ++i )
        {
            List<String> labelset = m_labels.get(i);
            if(labelset.contains(label))
                return m_inputUnits.get(i);
        }
        throw new RuntimeException("Cannot find unit");
    }

    private void assignLabels()
    {
        for ( int i = 0; i < m_labels.size(); ++i )
        {
            /* some input units may have multiple labels. That's why m_labels
             * is a List of List of Strings */
            final List<String> labelset = m_labels.get(i);
            if ( labelset.size() == 0 )
                continue;

            final Unit target = m_inputUnits.get(i);
            for ( final String label : labelset )
            {
                final List<UnitBox> boxes = m_labelToUnitBoxMap.get(label);
                if ( boxes == null )
                {
                    System.out.println( "Cannot find boxes for label.    This could be caused by classes other than the BytecodeLanguage using the assembler and is not a fatal error." );
                    continue;
                }
                for ( final UnitBox box : boxes )
                    box.setUnit( target );
            }
        }
    }

    Unit getLastUnitCreated() {
        return m_inputUnits.get(m_inputUnits.size()-1);
    }
}
