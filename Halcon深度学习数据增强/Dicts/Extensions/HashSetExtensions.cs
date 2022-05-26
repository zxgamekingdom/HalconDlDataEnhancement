using System.Collections.Generic;

namespace Halcon深度学习数据增强.Dicts.Extensions;

internal static class HashSetExtensions
{
    public static void AddRange<T>(this HashSet<T>? hashSet, IEnumerable<T> values)
    {
        foreach (var value in values) hashSet?.Add(value);
    }
}