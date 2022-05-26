using System;
using System.Collections.Generic;

namespace Halcon深度学习数据增强.Dicts.Extensions;

public static class CollectionExtensions
{
    public static void 检查重复<T>(this IEnumerable<T> values,
        ICollection<string> errorList,
        Func<T, string> 当重复时,
        IEqualityComparer<T>? comparer = default)
    {
        var set = new HashSet<T>(comparer);

        foreach (var value in values)
            if (set.Contains(value))
                errorList.Add(当重复时(value));
            else
                set.Add(value);
    }
}
