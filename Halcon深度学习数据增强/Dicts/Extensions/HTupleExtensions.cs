using HalconDotNet;

namespace Halcon深度学习数据增强.Dicts.Extensions;

internal static class HTupleExtensions
{

    public static HTuple? EmptyConvert(this HTuple t)
    {
        return t.Type == HTupleType.EMPTY ? null : t;
    }

}