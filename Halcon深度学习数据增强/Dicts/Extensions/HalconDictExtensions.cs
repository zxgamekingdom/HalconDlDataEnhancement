using System;
using HalconDotNet;

namespace Halcon深度学习数据增强.Dicts.Extensions;

internal static class HalconDictExtensions
{

    public static T? FromKeyTuple<T>(this HDict dict, string key, Func<HTuple, T> func)
    {
        var tuple = dict.GetDictTuple(key).EmptyConvert();

        return tuple == null ? default : func.Invoke(tuple);
    }

}
