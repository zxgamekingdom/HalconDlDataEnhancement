using HalconDotNet;

namespace TestProject;

public class UnitTestHalcon库
{

    [Fact]
    public void TestHTuple()
    {
        var h = new HTuple(2);

        // h.Length==1
        Assert.Equal(1, h.Length);

        // h.I==1
        Assert.Equal(2, h.I);

        // h.L==2
        Assert.Equal(2, h.L);
    }

    [Fact]
    public void Test空HTuple()
    {
        var tuple = new HTuple();

        // tuple.Type==HTupleType.EMPTY
        Assert.Equal(HTupleType.EMPTY, tuple.Type);

        // tuple.Length==0
        Assert.Equal(0, tuple.Length);
    }

}