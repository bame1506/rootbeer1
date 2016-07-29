
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.RootbeerGpu;


public class CountKernel implements Kernel
{
    /* each ref and long needs 4 Bytes, but for some reason.
     * The kernel seems to need 48 Bytes i.e. 16 bytes per member
     * Is it aligned after each variable ??
     * @see CUDAContext.java:writeBlocksList
     */
    private long[] mnHitsA;
    private long[] mnHitsB;
    private long   mnDiceRolls;

    public CountKernel
    (
        long[]     rnHitsA,
        long[]     rnHitsB,
        final long rnDiceRolls
    )
    {
        mnHitsA          = rnHitsA;
        mnHitsB          = rnHitsB;
        mnDiceRolls      = rnDiceRolls;
    }

    public void gpuMethod()
    {
        final int  randMax   = 0x7FFFFFFF;
        final long randMagic = 950706376;
        int dRandomSeed = 65163 + RootbeerGpu.getThreadId();

        assert( mnDiceRolls <= Integer.MAX_VALUE );
        final int dnDiceRolls = (int) mnDiceRolls;

        long nHitsA = 0;
        long nHitsB = 0;

        for ( int i = 0; i < dnDiceRolls; ++i )
        {
            dRandomSeed = (int)( (randMagic*dRandomSeed) % randMax );
            float x = (float) dRandomSeed / randMax;
            dRandomSeed = (int)( (randMagic*dRandomSeed) % randMax );
            float y = (float) dRandomSeed / randMax;

            if ( x*x + y*y < 1.0 )
                nHitsA += 1;
            else
                nHitsB += 1;
        }

        mnHitsA[ RootbeerGpu.getThreadId() ] = nHitsA;
        mnHitsB[ RootbeerGpu.getThreadId() ] = nHitsB;
    }
}
